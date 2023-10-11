import pandas as pd
from matplotlib import pyplot as plt

yolov8m = pd.read_csv('./yolov8m-seg/results.csv')
yolov8l = pd.read_csv('./yolov8l-seg/results.csv')
yolov8x = pd.read_csv('./yolov8x-seg/results.csv')



yolov8l = yolov8l[:121]
yolov8x = yolov8x[:121]
yolov8m = yolov8m[:121]

# get all spaces out of column names
yolov8m.columns = yolov8m.columns.str.replace(' ', '')
yolov8l.columns = yolov8l.columns.str.replace(' ', '')
yolov8x.columns = yolov8x.columns.str.replace(' ', '')


m_train_seg_loss = yolov8m['train/seg_loss']
l_train_seg_loss = yolov8l['train/seg_loss']
x_train_seg_loss = yolov8x['train/seg_loss']
m_val_seg_loss = yolov8m['val/seg_loss']
l_val_seg_loss = yolov8l['val/seg_loss']
x_val_seg_loss = yolov8x['val/seg_loss']


# plot train with color and label
plt.plot(m_train_seg_loss, color='green', label='yolov8m-train')
plt.plot(l_train_seg_loss, color='blue', label='yolov8l-train')
plt.plot(x_train_seg_loss, color='red', label='yolov8x-train')
plt.plot(m_val_seg_loss, color='green', alpha=0.5, label='yolov8m-val')
plt.plot(l_val_seg_loss, color='blue', alpha=0.5, label='yolov8l-val')
plt.plot(x_val_seg_loss, color='red', alpha=0.5, label='yolov8x-val')


# legend
plt.legend()
plt.show()

# save
plt.savefig('train_val_seg_loss.pdf')

plt.clf()


# mAp plot 'metrics/mAP50(M)', 'metrics/mAP50-95(M)'
m_map50 = yolov8m['metrics/mAP50(M)']
l_map50 = yolov8l['metrics/mAP50(M)']
x_map50 = yolov8x['metrics/mAP50(M)']
m_map5095 = yolov8m['metrics/mAP50-95(M)']
l_map5095 = yolov8l['metrics/mAP50-95(M)']
x_map5095 = yolov8x['metrics/mAP50-95(M)']

# plot map50 and map50-95 with color and label
plt.plot(m_map50, color='green', label='yolov8m-map50')
plt.plot(l_map50, color='blue', label='yolov8l-map50')
plt.plot(x_map50, color='red', label='yolov8x-map50')
plt.plot(m_map5095, color='green', alpha=0.5, label='yolov8m-map50-95')
plt.plot(l_map5095, color='blue', alpha=0.5, label='yolov8l-map50-95')
plt.plot(x_map5095, color='red', alpha=0.5, label='yolov8x-map50-95')

# legend
plt.legend()
plt.show()

# save
plt.savefig('map50_map5095.pdf')


# plot class loss 
m_train_class_loss = yolov8m['train/cls_loss']
l_train_class_loss = yolov8l['train/cls_loss']
x_train_class_loss = yolov8x['train/cls_loss']
m_val_class_loss = yolov8m['val/cls_loss']
l_val_class_loss = yolov8l['val/cls_loss']
x_val_class_loss = yolov8x['val/cls_loss']

# plot train with color and label
plt.plot(m_train_class_loss, color='green', label='yolov8m-train')
plt.plot(l_train_class_loss, color='blue', label='yolov8l-train')
plt.plot(x_train_class_loss, color='red', label='yolov8x-train')
plt.plot(m_val_class_loss, color='green', alpha=0.5, label='yolov8m-val')
plt.plot(l_val_class_loss, color='blue', alpha=0.5, label='yolov8l-val')
plt.plot(x_val_class_loss, color='red', alpha=0.5, label='yolov8x-val')





