"""Render scan_run2 joint poses with measured wrist calibration and marker geometry."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import ast
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen


def quat(r):
    q = Rotation.from_matrix(r).as_quat()
    return ' '.join(map(str, q[[3,0,1,2]]))


def main():
    root = Path('real2sim/zed_capture/scan_run2')
    out = Path('figs/sim2real/scan_run2_wrist_comparison')
    out.mkdir(parents=True, exist_ok=True)
    sol = json.loads(Path('real2sim/scan_pipeline/ground_run2_solved.json').read_text())
    report = json.loads(Path('figs/sim2real/wrist_calibrated_corrected/report.json').read_text())
    mount = np.array(report['camera']['T_sim_hand_camera_opencv'])
    intr = json.loads((root/'intrinsics.json').read_text())['zedm']['left']
    robot = Panda(idn=0)
    robot.add_gripper(gripper_factory('PandaGripper', idn=0))
    xml = ET.fromstring(robot.get_xml())
    visual = xml.find('visual')
    if visual is None:
        visual = ET.SubElement(xml, 'visual')
    ET.SubElement(visual, 'headlight', ambient='0.8 0.8 0.8', diffuse='0.4 0.4 0.4', specular='0 0 0', active='1')
    world = xml.find('worldbody')
    asset = xml.find('asset')
    hand = world.find(".//body[@name='robot0_right_hand']")
    camera = hand.find("camera[@name='robot0_eye_in_hand']")
    # Render an oversized centered pinhole and crop to the measured principal point.
    scale = 1/3
    width, height = 736, 414
    rw, rh = 800, 440
    camera.set('pos', ' '.join(map(str,mount[:3,3])))
    camera.set('quat', quat(mount[:3,:3] @ np.diag([1,-1,-1])))
    camera.set('fovy', str(np.degrees(2*np.arctan(rh/(2*intr['fy']*scale)))))
    board_t = np.array(sol['T_base_board'])
    board = ET.SubElement(world,'body',name='measured_board',pos=' '.join(map(str,board_t[:3,3])),quat=quat(board_t[:3,:3]))
    points = {ast.literal_eval(k): np.array(v) for k,v in json.loads(Path('real2sim/scan_pipeline/board_model.json').read_text()).items()}
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    # Marker cells are planar triangles interpolated through the measured corners.
    for marker in sorted(set(k[0] for k in points)):
        corners = np.array([points[marker,j] for j in range(4)])
        pattern = cv2.aruco.generateImageMarker(dictionary, marker, 7)
        for row in range(7):
            for col in range(7):
                vertices=[]
                for u,v in [(col/7,row/7),((col+1)/7,row/7),((col+1)/7,(row+1)/7),(col/7,(row+1)/7)]:
                    vertices.append((1-u)*(1-v)*corners[0]+u*(1-v)*corners[1]+u*v*corners[2]+(1-u)*v*corners[3])
                # Thin solid prism avoids a coplanar mesh compiler failure.
                vertices=np.array(vertices)
                vertices=np.concatenate([vertices,vertices+[0,0,-0.0002]])
                name=f'm{marker}_{row}_{col}'
                ET.SubElement(asset,'mesh',name=name,vertex=' '.join(map(str,vertices.ravel())),face='0 1 2 0 2 3 4 6 5 4 7 6 0 4 5 0 5 1 1 5 6 1 6 2 2 6 7 2 7 3 3 7 4 3 4 0')
                color='0.04 0.04 0.04 1' if pattern[row,col]==0 else '0.95 0.95 0.95 1'
                ET.SubElement(board,'geom',type='mesh',mesh=name,rgba=color,contype='0',conaffinity='0',group='1')
    ET.SubElement(world,'light',pos='0 0 2',dir='0 0 -1',diffuse='0.8 0.8 0.8',ambient='0.5 0.5 0.5')
    sim=MjSim.from_xml_string(ET.tostring(xml,encoding='unicode'))
    ctx=MjRenderContextOffscreen(sim,device_id=0)
    ctx.vopt.geomgroup[0]=0
    ctx.vopt.geomgroup[1]=1
    ctx.vopt.sitegroup[:]=0
    rows=[]
    for folder in sorted(root.glob('pose_*')):
        state=json.loads((folder/'state.json').read_text())
        for j,q in enumerate(state['joints']):
            sim.data.set_joint_qpos(f'robot0_joint{j+1}',q)
        panels=[Image.open(folder/'zedm_left.png').convert('RGB').resize((width,height))]
        for gap in (0.0,0.04):
            sim.data.set_joint_qpos('gripper0_finger_joint1',gap)
            sim.data.set_joint_qpos('gripper0_finger_joint2',-gap)
            sim.forward()
            frame=sim.render(width=rw,height=rh,camera_name='robot0_eye_in_hand')[::-1]
            x=rw/2-intr['cx']*scale
            y=rh/2-intr['cy']*scale
            frame=cv2.warpAffine(frame,np.array([[1,0,-x],[0,1,-y]],dtype=float),(width,height))
            panels.append(Image.fromarray(frame))
        strip=Image.new('RGB',(width*3,height+24),'white')
        for j,panel in enumerate(panels):
            strip.paste(panel,(j*width,24))
        ImageDraw.Draw(strip).text((5,5),folder.name+' | REAL wrist | SIM closed | SIM open (opening not recorded)',fill='black')
        strip.save(out/(folder.name+'.png'))
        if folder.name in ('pose_000','pose_007','pose_013'):
            rows.append(strip)
        print(folder.name,flush=True)
    sheet=Image.new('RGB',(width*3,(height+24)*len(rows)),'white')
    for i,row in enumerate(rows):
        sheet.paste(row,(0,i*(height+24)))
    sheet.save(out/'comparison.png')
    (out/'notes.json').write_text(json.dumps({'poses':14,'intrinsics':'native scaled 1/3, principal point included','gripper_opening':'unknown; closed and 80mm total opening shown','scene':'measured marker geometry only; no real background texture','mount':mount.tolist()},indent=2))


if __name__=='__main__':
    main()
