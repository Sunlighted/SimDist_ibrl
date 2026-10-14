"""Fixed exposure / gain / white balance for ZED cameras on plain V4L2 (no ZED SDK).

Auto-exposure makes the image brightness depend on what is in view (the dark gripper filling
the wrist view, the arm entering the scene), which the sim never does. The settings come from
one file, camera_controls.json next to this script, so recording, calibration tools and
deployment all capture with the same values.

Exposure and gain are not standard UVC controls on the ZED: they are sensor registers behind
Stereolabs' UVC extension unit (port of stereolabs/zed-open-capture, src/videocapture.cpp, BSD):
XU unit 4 / selector 2, 384-byte vendor buffer; AEC/AGC = bit 1 of ISP_CTRL_{LEFT,RIGHT};
exposure = sensor registers 0x3500-0x3502, gain = 0x3507-0x3509, sensor 0 = left, 1 = right.
White balance is a standard V4L2 control.

Facts measured on our ZED 2i / ZED-M (2026-10-14):
  * the sensor registers only answer while the camera is STREAMING (reads give 0xFF otherwise);
  * every stream (re)open resets the exposure (to ~777 raw at HD720@30, heavily over-exposed)
    while AEC stays off => apply() must run after EVERY open, once a frame has been read;
  * a USB replug turns AEC back on;
  * the raw exposure unit depends on the fps (max 1100 @30, 880 @60), so the values below are
    for the fps in the file;
  * white_balance_temperature is coarse: 3000-3300 give the same image, 3400-3500 another;
  * ⚠️ writing white_balance_red_component on the ZED-M hung its stream until a USB replug
    (uvcvideo "Failed to set UVC probe control: -71"). Never touch the *_component controls.

CLI:  python3 zed_controls.py status|apply|auto [--cams ZED-M "ZED 2i"]
      (the camera must be streaming in another process for status / apply)
"""
import argparse
import ctypes
import fcntl
import json
import os
import struct
import time
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parent / "camera_controls.json"

UVCIOC_CTRL_QUERY = 0xC0107521          # _IOWR('u', 0x21, struct uvc_xu_control_query)
VIDIOC_S_CTRL = 0xC008561C              # _IOWR('V', 28, struct v4l2_control)
VIDIOC_G_CTRL = 0xC008561B
UVC_SET_CUR, UVC_GET_CUR, UVC_GET_LEN = 0x01, 0x81, 0x85
XU_UNIT, XU_SELECTOR = 0x04, 0x02
XU_TASK_SET, XU_TASK_GET = 0x50, 0x51
ISP_CTRL = {0: 0x80181033, 1: 0x80181833}
AEC_AGC_BIT = 0x02
ADDR_EXP = (0x3500, 0x3501, 0x3502)
ADDR_GAIN = (0x3507, 0x3508, 0x3509)
EXP_RAW_MAX = {15: 1550, 30: 1100, 60: 880, 100: 720}
EXP_RAW_MIN = 2
CID_WB_AUTO, CID_WB_TEMP = 0x0098090C, 0x0098091A


def load(path=DEFAULT_FILE):
    return json.load(open(path))


class ZedControls:
    def __init__(self, device):
        """device: /dev/videoN path or index. Opens its own fd (works next to a streaming one)."""
        self.path = device if str(device).startswith("/dev/") else f"/dev/video{device}"
        self.fd = os.open(self.path, os.O_RDWR)
        n = self._xu(UVC_GET_LEN, bytearray(2))
        self.len = n[0] | (n[1] << 8)                         # 384 on USB 3, 64 on USB 2

    def close(self):
        os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- raw access
    def _xu(self, query, buf):
        cbuf = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
        q = struct.pack("BBBxH2xQ", XU_UNIT, XU_SELECTOR, query, len(buf), ctypes.addressof(cbuf))
        fcntl.ioctl(self.fd, UVCIOC_CTRL_QUERY, bytearray(q))
        return buf

    def _vendor(self, buf, read):
        self._xu(UVC_SET_CUR, buf)
        time.sleep(0.0003)
        if read:
            self._xu(UVC_GET_CUR, buf)
        return buf

    def _hdr(self, task, cmd, b2, aw, address):
        b = bytearray(self.len)
        b[0:5] = bytes((task, cmd, b2, aw, 0x01))
        b[5:9] = address.to_bytes(4, "big")
        b[9:13] = bytes((0x00, 0x01, 0x00, 0x01))
        return b

    def _sys(self, address, value=None):
        b = self._hdr(XU_TASK_GET if value is None else XU_TASK_SET, 0xA2, 0, 0x04, address)
        if value is not None:
            b[16] = value & 0xFF
        return self._vendor(b, value is None)[17]

    def _sensor(self, side, address, value=None):
        b = self._hdr(XU_TASK_GET if value is None else XU_TASK_SET, 0xA3 if side == 0 else 0xA5,
                      0x6C, 2, address)
        b[9] = 0x90                                            # limit 1 + page address flags
        if value is not None:
            b[16] = value & 0xFF
        return self._vendor(b, value is None)[17]

    def _ctrl(self, cid, value=None):
        buf = bytearray(struct.pack("Ii", cid, 0 if value is None else int(value)))
        fcntl.ioctl(self.fd, VIDIOC_G_CTRL if value is None else VIDIOC_S_CTRL, buf)
        return struct.unpack("Ii", buf)[1]

    # ---- high level (both sensors)
    def aecagc(self):
        return [bool(self._sys(ISP_CTRL[s]) & AEC_AGC_BIT) for s in (0, 1)]

    def set_aecagc(self, on):
        for s in (0, 1):
            v = self._sys(ISP_CTRL[s])
            self._sys(ISP_CTRL[s], (v | AEC_AGC_BIT) if on else (v & ~AEC_AGC_BIT & 0xFF))

    def exposure_raw(self):
        out = []
        for s in (0, 1):
            h, m, l = (self._sensor(s, a) for a in ADDR_EXP)
            out.append((h << 12) | (m << 4) | (l >> 4))
        return out

    def set_exposure_raw(self, raw):
        raw = max(EXP_RAW_MIN, int(raw))
        for s in (0, 1):
            for a, v in zip(ADDR_EXP, ((raw >> 12) & 0xFF, (raw >> 4) & 0xFF, (raw << 4) & 0xF0)):
                self._sensor(s, a, v)

    def gain_raw(self):
        return [(self._sensor(s, ADDR_GAIN[1]) << 8) | self._sensor(s, ADDR_GAIN[2]) for s in (0, 1)]

    def set_gain_raw(self, raw):
        for s in (0, 1):
            self._sensor(s, ADDR_GAIN[0], 0)
            self._sensor(s, ADDR_GAIN[1], (int(raw) >> 8) & 0xFF)
            self._sensor(s, ADDR_GAIN[2], int(raw) & 0xFF)

    def white_balance(self):
        return {"auto": self._ctrl(CID_WB_AUTO), "temperature": self._ctrl(CID_WB_TEMP)}

    def set_white_balance(self, temperature):
        self._ctrl(CID_WB_AUTO, 0)                           # temperature is inactive while auto
        self._ctrl(CID_WB_TEMP, temperature)

    def status(self):
        return {"aecagc": self.aecagc(), "exposure_raw": self.exposure_raw(),
                "gain_raw": self.gain_raw(), **{"wb_" + k: v for k, v in self.white_balance().items()}}


def apply(device, settings=None, fps=None, grab=None, tries=5):
    """Lock exposure / gain / white balance on a camera that is streaming RIGHT NOW (call it
    after every open). `grab` reads one frame from that stream; it is needed because the sensor
    reads 0 for the first ~3 frames after an open and then its own init writes the exposure
    (~771 raw), overwriting anything written before. So: wait until the sensor answers, write,
    stream a few frames, read back; retry if the init overwrote it. Returns the status."""
    s = settings if settings is not None else load()
    if fps is not None and int(fps) != int(s["fps"]):
        raise ValueError(f"camera_controls.json is for {s['fps']} fps, the stream is {fps} fps: "
                         f"the raw exposure unit differs")
    want = dict(exposure_raw=[s["exposure_raw"]] * 2, gain_raw=[s["gain_raw"]] * 2)
    pump = grab if grab is not None else (lambda: time.sleep(1 / 30))
    with ZedControls(device) as c:
        for _ in range(60):                                   # sensor ready: a real value
            if all(0 < e < 0xFFFFF for e in c.exposure_raw()):
                break
            pump()
        for _ in range(tries):
            c.set_aecagc(False)
            c.set_exposure_raw(s["exposure_raw"])
            c.set_gain_raw(s["gain_raw"])
            c.set_white_balance(s["wb_temperature"])
            for _ in range(4):
                pump()
            st = c.status()
            if (not any(st["aecagc"]) and st["wb_auto"] == 0
                    and all(st[k] == v for k, v in want.items())):
                return st
    raise RuntimeError(f"{device}: controls did not stick after {tries} tries: {st}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "apply", "auto"])
    ap.add_argument("--cams", nargs="+", default=["ZED-M", "ZED 2i"], help="V4L2 name prefixes")
    ap.add_argument("--file", default=str(DEFAULT_FILE))
    a = ap.parse_args()
    for model in a.cams:
        dev = next(i for i in range(16) if os.path.exists(f"/sys/class/video4linux/video{i}/name")
                   and open(f"/sys/class/video4linux/video{i}/name").read().startswith(model))
        if a.cmd == "apply":
            print(model, apply(dev, load(a.file)))
        else:
            with ZedControls(dev) as c:
                if a.cmd == "auto":
                    c.set_aecagc(True)
                    c._ctrl(CID_WB_AUTO, 1)
                print(model, c.status())


if __name__ == "__main__":
    main()
