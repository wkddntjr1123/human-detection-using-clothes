from collections import defaultdict
import os,json
from time import time,sleep
from color import check_color
from segment import segmentation_frame
import torch
import cv2
from threading import Thread
from collections import deque
import asyncio         
import websockets    
import networks

"""GLOBAL SETTING"""
# yolov5 로드
model = torch.hub.load(
    "yolov5",
    "custom",
    path=os.path.join("yolov5", "person_small.pt"),
    source="local",
)
model.conf = 0.6  # confidence 값 설정
# segment 모델 로드
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num_classes = 7
label = ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs']
seg_model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
state_dict = torch.load("pascal.pth")['state_dict']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
seg_model.load_state_dict(new_state_dict)
seg_model.cuda()
seg_model.eval()

# 동기화 큐
queue = deque()
stopped = False

"""END GLOBAL SETTING"""

def update(cap,out):
    cnt = 0
    while True:
        global stopped
        fps = cap.get(cv2.CAP_PROP_FPS)
        cnt +=1
        if stopped:
            return
        if not (len(queue) >= 200) :
            grabbed = cap.grab()
            if cnt % round(fps/2) != 0 : # 0.5초 단위로 동기화 큐에 넣기
                continue
            if not grabbed:
                stopped=True
                cap.release()
                out.release()
                return
            queue.append(cap.retrieve()[1])
def more():
    return len(queue) > 0

def read():
    return queue.popleft()

# yolov5로 사람 객체 찾기
# INPUT : frame
# OUTPUT : (사람객체BOX좌표 리스트, 사람객체개수)
def score_frame(frame):
    global model
    model.to("cuda")
    frame = [frame]
    results = model(frame)
    cord = results.xyxyn[0][:, :-1].cpu().numpy()
    n = len(results.xyxyn[0][:, -1].cpu().numpy())             
    return cord, n

# 1. score_frame으로 사람 객체 있는지 판단 => 없다면 frame 그냥 리턴
# 2. 있다면 segmentation하여 상하의 map 생성
# 3. 각 인물의 상하의 색을 판단
# 4. 상하의색 일치하면 box를 그린 frame을 리턴
def resolve_frame(frame,resize_cnt,top,bot,top_color,bot_color):
    cord, n = score_frame(frame)
    draw_spots = []
    if n == 0 : # 사람 검출 안되면 바로 frame 리턴
        #cv2.imwrite(os.path.join(BASE_DIR, "static", f"{self.frame_cnt}.jpg"), frame)
        return draw_spots
    map_result = segmentation_frame(frame,seg_model)
    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) # hsv색공간으로 변환
    for i in range(n): # 각 사람 박스를  순회
        row = cord[i]
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
        found = check_color(hsv_frame,top,bot,top_color,bot_color,map_result,x1,y1,x2,y2) #상하의 모두 선택한 색과 일치하면 True
        if found :
            draw_spots.append((x1*(2**resize_cnt),y1*(2**resize_cnt),x2*(2**resize_cnt),y2*(2**resize_cnt))) # 축소했던만큼 다시 2배수를 한 위치에 박스 그려야함
    #cv2.imwrite(os.path.join(BASE_DIR, "static", f"{self.frame_cnt}.jpg"), frame)
    return draw_spots


# 웹소켓 실행 로직
async def accept(websocket, path):
    global stopped,queue
    is_top=is_bot=input_top_color=input_bot_color=None # 상의,하의,상의색,하의색
    while True:
        message = await websocket.recv() # 클라이언트로부터 메시지를 대기
        if message == "START" : 
            await websocket.send("JSON")
        elif message == "JSON" :
            message = await websocket.recv()
            message = json.loads(message)
            is_top = message["top"]
            is_bot = message["bottom"]
            input_top_color = message["input_top_color"]
            input_bot_color = message["input_bot_color"]
            print("JSON INPUT: ",is_top,is_bot,input_top_color,input_bot_color)
            await websocket.send("FILE")
        elif message == "FILE" :
            file = await websocket.recv()
            print("FILE:",type(file))
            FILE_NAME = "input.mp4"
            if os.path.isfile(FILE_NAME) :
                os.remove(FILE_NAME)
            with open(FILE_NAME,"wb") as out_file:
                out_file.write(file)
            cap = cv2.VideoCapture(FILE_NAME)
            x_shape = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 프레임 가로 크기
            y_shape = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 세로 크기
            resize_cnt = 0
            max_size = 256 # 128은 결과가 너무 줄어들어버림 512는 너무 속도가 안나옴
            while x_shape > max_size and y_shape > max_size : # 이미지 크기 줄여서 시간 최적화
                x_shape//=2
                y_shape//=2
                resize_cnt+=1
            total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 총 프레임 수
            fps = cap.get(cv2.CAP_PROP_FPS) # 초당 프레임 수
            await websocket.send(f"TIME : {str(total_frame/fps)}") # 클라이언트에게 파일 전송 완료됨을 알림 : 총 재생시간 전송
            """결과 동영상 저장을 위함"""
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter("output.mp4", codec, round(fps), (x_shape, y_shape))  # 프레임을 동영상으로 저장

            cur = 0 # 진행중인 프레임 번호
            flush_cnt = 0
            read_thread = Thread(target=update,args=(cap,out)) # frame read용 쓰레드
            read_thread.daemon = True
            read_thread.start()
            result = defaultdict(list)
            start = time()
            while True :
                if stopped and not queue:
                    break
                if not queue:
                    sleep(0.01)
                    continue
                frame = read()  # 동영상에서 한 프레임 가져오기
                cur += fps/2
                flush_cnt += 1
                if flush_cnt==10: # 5초 단위로 분석된 정보를 front로 전송
                    print(result)
                    flush_cnt = 0
                    if result :
                        await websocket.send(json.dumps(result))
                        result = defaultdict(list)
                print(f"{cur} / {total_frame}") # 진행상황 print
                spots = resolve_frame(cv2.resize(frame,dsize=(x_shape,y_shape)),resize_cnt,is_top,is_bot,input_top_color,input_bot_color) # 색 일치하는 사람에만 box표시, 결과 있으면 spots리스트에 box좌표 담긴 상태
                if len(spots):
                    cur_time = round(cur/fps,1) # 객체 발견 시간
                    result[cur_time] = spots # spots는 [(x1,y1,x2,y2),(x1,y1,x2,y2),(x1,y1,x2,y2),...]
                    for x1,y1,x2,y2 in spots:
                        bgr = (0, 255, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 4) # 색 일치하면 frame에서 해당 사람에 box표시
                        cv2.imwrite(os.path.join("outputs",f"{round(cur,1)}.jpg"),frame)
            print(f" 걸린시간 : {time()-start}")
            print(result)
            await websocket.send(json.dumps(result)) # 마지막 result 전송
            await websocket.send("END") # 모든 프로세스가 종료. 클라이언트에서 연결 끊도록 메시지
            queue = deque()
            stopped = False

# 서버 시작 후 비동기로 대기
start_server = websockets.serve(accept, "0.0.0.0", 8000,max_size=2000000000)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()