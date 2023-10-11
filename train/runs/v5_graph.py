import pandas as pd
from matplotlib import pyplot as plt

yolov8m = pd.read_csv('./yolov5m-seg/results.csv')
yolov8l = pd.read_csv('./yolov5l-seg/results.csv')
yolov8x = pd.read_csv('./yolov5x-seg/results.csv')

# get all spaces out of column names
yolov8m.columns = yolov8m.columns.str.replace(' ', '')
yolov8l.columns = yolov8l.columns.str.replace(' ', '')
yolov8x.columns = yolov8x.columns.str.replace(' ', '')

m_train_seg_loss = yolov8m['train/seg_loss'][:-1] 
l_train_seg_loss = yolov8l['train/seg_loss'][:-1]
x_train_seg_loss = yolov8x['train/seg_loss'][:-1]
m_val_seg_loss = yolov8m['val/seg_loss'][:-1]
l_val_seg_loss = yolov8l['val/seg_loss'][:-1]
x_val_seg_loss = yolov8x['val/seg_loss'][:-1]

# Calculate the moving average for 'train/seg_loss' with a window size of 5
window_size = 5
m_train_seg_loss_ma = m_train_seg_loss.rolling(window=window_size, min_periods=1).mean()
l_train_seg_loss_ma = l_train_seg_loss.rolling(window=window_size, min_periods=1).mean()
x_train_seg_loss_ma = x_train_seg_loss.rolling(window=window_size, min_periods=1).mean()
m_val_seg_loss_ma = m_val_seg_loss.rolling(window=window_size, min_periods=1).mean()
l_val_seg_loss_ma = l_val_seg_loss.rolling(window=window_size, min_periods=1).mean()
x_val_seg_loss_ma = x_val_seg_loss.rolling(window=window_size, min_periods=1).mean()

# plot the moving average
plt.plot(m_train_seg_loss_ma, color='green', label='yolov5m-train')
plt.plot(l_train_seg_loss_ma, color='blue', label='yolov5l-train')
plt.plot(x_train_seg_loss_ma, color='red', label='yolov5x-train')
plt.plot(m_val_seg_loss_ma, color='green', alpha=0.5, label='yolov5m-val')
plt.plot(l_val_seg_loss_ma, color='blue', alpha=0.5, label='yolov5l-val')
plt.plot(x_val_seg_loss_ma, color='red', alpha=0.5, label='yolov5x-val')

# legend
plt.legend()
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Segmentation Loss')
plt.xlim(0)
plt.savefig('v5_train_val_seg_loss.pdf')
plt.show()

# save
plt.clf()

# mAp plot 'metrics/mAP_0.5(M),metrics/mAP_0.5:0.95(M)'
m_map50 = yolov8m['metrics/mAP_0.5(M)'][:-1]
l_map50 = yolov8l['metrics/mAP_0.5(M)'][:-1]
x_map50 = yolov8x['metrics/mAP_0.5(M)'][:-1]
m_map5095 = yolov8m['metrics/mAP_0.5:0.95(M)'][:-1]
l_map5095 = yolov8l['metrics/mAP_0.5:0.95(M)'][:-1]
x_map5095 = yolov8x['metrics/mAP_0.5:0.95(M)'][:-1]

# Calculate the moving average for map50 and map50-95
m_map50_ma = m_map50.rolling(window=window_size, min_periods=1).mean()
l_map50_ma = l_map50.rolling(window=window_size, min_periods=1).mean()
x_map50_ma = x_map50.rolling(window=window_size, min_periods=1).mean()
m_map5095_ma = m_map5095.rolling(window=window_size, min_periods=1).mean()
l_map5095_ma = l_map5095.rolling(window=window_size, min_periods=1).mean()
x_map5095_ma = x_map5095.rolling(window=window_size, min_periods=1).mean()

# plot map50 and map50-95 with color and label
plt.plot(m_map50_ma, color='green', label='yolov5m-map50')
plt.plot(l_map50_ma, color='blue', label='yolov5l-map50')
plt.plot(x_map50_ma, color='red', label='yolov5x-map50')
plt.plot(m_map5095_ma, color='green', alpha=0.5, label='yolov5m-map50-95')
plt.plot(l_map5095_ma, color='blue', alpha=0.5, label='yolov5l-map50-95')
plt.plot(x_map5095_ma, color='red', alpha=0.5, label='yolov5x-map50-95')

# legend
plt.legend()
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('mAP')
plt.xlim(0)
plt.savefig('v5_map50_map5095.pdf')
plt.show()

# save
plt.clf()

# plot class loss
m_train_class_loss = yolov8m['train/cls_loss'][:-1]
l_train_class_loss = yolov8l['train/cls_loss'][:-1]
x_train_class_loss = yolov8x['train/cls_loss'][:-1]
m_val_class_loss = yolov8m['val/cls_loss'][:-1]
l_val_class_loss = yolov8l['val/cls_loss'][:-1]
x_val_class_loss = yolov8x['val/cls_loss'][:-1]

# Calculate the moving average for train and val class loss
m_train_class_loss_ma = m_train_class_loss.rolling(window=window_size, min_periods=1).mean()
l_train_class_loss_ma = l_train_class_loss.rolling(window=window_size, min_periods=1).mean()
x_train_class_loss_ma = x_train_class_loss.rolling(window=window_size, min_periods=1).mean()
m_val_class_loss_ma = m_val_class_loss.rolling(window=window_size, min_periods=1).mean()
l_val_class_loss_ma = l_val_class_loss.rolling(window=window_size, min_periods=1).mean()
x_val_class_loss_ma = x_val_class_loss.rolling(window=window_size, min_periods=1).mean()

# plot train with color and label
plt.plot(m_train_class_loss_ma, color='green', label='yolov5m-train')
plt.plot(l_train_class_loss_ma, color='blue', label='yolov5l-train')
plt.plot(x_train_class_loss_ma, color='red', label='yolov5x-train')
plt.plot(m_val_class_loss_ma, color='green', alpha=0.5, label='yolov5m-val')
plt.plot(l_val_class_loss_ma, color='blue', alpha=0.5, label='yolov5l-val')
plt.plot(x_val_class_loss_ma, color='red', alpha=0.5, label='yolov5x-val')

# legend
plt.legend()
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Class Loss')
plt.xlim(0)
plt.savefig('v5_train_val_class_loss.pdf')
plt.show()
