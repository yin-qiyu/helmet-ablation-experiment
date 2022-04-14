# 精确率（precision）、召回率（recall）、准确率（accuracy）

- [ ] [博客](https://blog.csdn.net/duan19920101/article/details/121726392)

![image-20220413160108001](https://raw.githubusercontent.com/yin-qiyu/picbed/master/img/202204131601052.png)



# 拟合

收敛后验证loss和训练loss 收敛后的loss 就三种情况，完全重合
或者是一个上一个下（未收之前不参考）收敛后可以分析
若验证在上，说明，没有达到预期，过拟合，模型复杂度大于数据复杂度，所以数据少，做数据增广，或者模型减小。
若训练在上，说明，没有达到预期，欠拟合，模型复杂度低于数据复杂度，提高模型复杂度。
- [ ] 原文链接：https://blog.csdn.net/weixin_32759777/article/details/122022491



# 结果解读

- [ ] [yolov5 训练结果解析_高祥xiang的博客-CSDN博客_yolov5训练结果](https://blog.csdn.net/qq_27278957/article/details/119968555?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.pc_relevant_antiscanv2&spm=1001.2101.3001.4242.1&utm_relevant_index=3)



- [ ] [【机器学习】PRC（PR曲线）_littlemichelle的博客-CSDN博客_pr曲线](https://blog.csdn.net/weixin_31866177/article/details/88776718?ops_request_misc=%7B%22request%5Fid%22%3A%22162484280716780261910347%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=162484280716780261910347&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-7-88776718.pc_search_result_before_js&utm_term=PR曲线&spm=1018.2226.3001.4187)

  

- [ ] [PR曲线_qq_39114535的博客-CSDN博客_pr曲线平衡点](https://blog.csdn.net/qq_39114535/article/details/115006696?ops_request_misc=%7B%22request%5Fid%22%3A%22162484280716780269893500%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fall.%22%7D&request_id=162484280716780269893500&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-9-115006696.pc_search_result_before_js&utm_term=PR曲线&spm=1018.2226.3001.4187)







# 调参经验

- [ ] [win10+anaconda+pychram+yolov5调参、训练经验慢慢更新_D1M0N172的博客-CSDN博客_yolov5调参](https://blog.csdn.net/qwazp3526cn/article/details/115436153?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-6.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-6.pc_relevant_antiscanv2&utm_relevant_index=9)





- [ ] [目标检测 YOLOv5 - 损失函数的改进_TheOldManAndTheSea的博客-CSDN博客_yolov5修改损失函数](https://blog.csdn.net/flyfish1986/article/details/120534863?ops_request_misc=%7B%22request%5Fid%22%3A%22164994154016780265435447%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fall.%22%7D&request_id=164994154016780265435447&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-19-120534863.142^v8^pc_search_result_cache,157^v4^control&utm_term=yolov5+loss&spm=1018.2226.3001.4187)







# 小目标优化

- [ ] [YOLOV5 模型和代码修改——针对小目标识别_xiaoY322的博客-CSDN博客_yolov5小目标](https://blog.csdn.net/weixin_56184890/article/details/119840555?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.pc_relevant_default&spm=1001.2101.3001.4242.1&utm_relevant_index=3)

- [ ] [YOLOV5 的小目标检测网络结构优化方法汇总(附代码)_3Ｄ视觉工坊的博客-CSDN博客](https://blog.csdn.net/Yong_Qi2015/article/details/122375061?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.pc_relevant_default&spm=1001.2101.3001.4242.1&utm_relevant_index=3)