# Garbage Detection Application
## Author: Ayush Tripathi (atripathi7783@gmail.com)


Goals:
- Detect and classify image with the correct bin to sort it in
- Have high accuracy on classification
- Be able to do so through a live video feed



## Plan
1. Write YOLO and Mobile-Net Architecture to be compatible with new data
2. Train off of TacoDataset, choose the best time running metrics and computationally efficient (use 7 classes for output, define more directly internally)
3. Work on frame-process.py, have it so that it can use the model to classify realtime
4. Create front end process 
5. Wrap into executable(if possible)