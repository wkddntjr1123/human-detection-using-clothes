# 사용자 입력에 따른 특정 사람 인식

## 필요 가중치 파일 : semantic segmentation을 위한 pascal.pth, yolov5를 위한 person.pt

### 성능 개선 방안

1. 멀티 스레드를 이용한 비디오 I/O 및 프레임 처리 파이프 라인 구축
2. FPS/2 지점만 버퍼 큐에 넣기
3. semantic segmentation 전에 frame resize를 통한 픽셀 병합
4. 프레임 처리 결과를 서버에서 재구축하는게 아니라, 웹소켓을 통해 5초단위로 JSON 결과를 전송. 클라이언트에서는 JSON에 있는 좌표에 Div를 통해 박스 그리기
