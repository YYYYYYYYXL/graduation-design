# BRISQUE 评分文件

此目录仅用于无参考图像质量评分，与五个去噪网络的权重分开存放。
评分文件已随项目保存，网页运行期间不下载、不训练。

文件取自 OpenCV contrib 4.11.0 的 quality 模块：

- [brisque_model_live.yml](https://github.com/opencv/opencv_contrib/blob/4.11.0/modules/quality/samples/brisque_model_live.yml)
- [brisque_range_live.yml](https://github.com/opencv/opencv_contrib/blob/4.11.0/modules/quality/samples/brisque_range_live.yml)
- [OpenCV 许可证](https://github.com/opencv/opencv_contrib/blob/4.11.0/LICENSE)，副本保存在本目录 LICENSE。
- [接口与评分说明](https://docs.opencv.org/4.11.0/d8/d99/classcv_1_1quality_1_1QualityBRISQUE.html)

BRISQUE 根据自然图像统计估计感知质量，使用 LIVE-R2 评分模型。当前实现返回 0～100，越低通常表示质量越好；不是准确率，也不能证明细节还原正确。对同一张图片的输入、输出使用相同尺寸，不为评分再次缩放。

研究引用：A. Mittal, A. K. Moorthy and A. C. Bovik, “No-Reference Image Quality Assessment in the Spatial Domain”, IEEE Transactions on Image Processing, 2012. 软件来源为 University of Texas at Austin 的 Laboratory for Image and Video Engineering (LIVE) 与 Center for Perceptual Systems (CPS)。
