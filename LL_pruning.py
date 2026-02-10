from ultralytics import YOLO
import torch
from ultralytics.nn.modules import Bottleneck, Conv, C2f, SPPF, Detect
import os
import torch.nn as nn

# 尝试导入 RFAConv 以便进行类型检查，如果找不到文件则忽略（依靠属性判断）
try:
    from RFAConv import RFAConv
except ImportError:
    pass


class PRUNE():
    def __init__(self) -> None:
        self.threshold = None

    def get_threshold(self, model, factor=0.8):
        ws = []
        bs = []
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                w = m.weight.abs().detach()
                b = m.bias.abs().detach()
                ws.append(w)
                bs.append(b)
                # print(name, w.max().item(), w.min().item(), b.max().item(), b.min().item())
                # print()
        # keep
        ws = torch.cat(ws)
        self.threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]

    def prune_conv(self, conv1, conv2):
        # ------------------------------------------------------
        # FIX START: 兼容 RFAConv 或其他自定义模块
        # 如果 conv1 是 RFAConv，它没有 .bn，但它的 conv1.conv 属性里有 .bn
        # 我们定位到真正包含 bn 和 conv 权重的那个子模块
        # ------------------------------------------------------
        target_module = conv1

        # 情况1: 是 RFAConv 这种封装层 (没有 bn, 但有 conv 属性且 conv 属性里有 bn)
        if not hasattr(target_module, 'bn') and hasattr(target_module, 'conv'):
            target_module = target_module.conv

        # 如果经过上面处理还是找不到 bn，说明结构不支持，直接跳过或报错
        if not hasattr(target_module, 'bn'):
            print(f"Warning: Skipping module {type(conv1)} because 'bn' layer not found.")
            return

        ## a. 根据BN中的参数，获取需要保留的index================
        gamma = target_module.bn.weight.data.detach()
        beta = target_module.bn.bias.data.detach()

        keep_idxs = []
        local_threshold = self.threshold
        while len(keep_idxs) < 8:  ## 若剩余卷积核<8, 则降低阈值重新筛选
            keep_idxs = torch.where(gamma.abs() >= local_threshold)[0]
            local_threshold = local_threshold * 0.5
        n = len(keep_idxs)
        print(f"Pruning ratio: {n / len(gamma) * 100:.2f}%")

        ## b. 利用index对BN进行剪枝============================
        target_module.bn.weight.data = gamma[keep_idxs]
        target_module.bn.bias.data = beta[keep_idxs]
        target_module.bn.running_var.data = target_module.bn.running_var.data[keep_idxs]
        target_module.bn.running_mean.data = target_module.bn.running_mean.data[keep_idxs]
        target_module.bn.num_features = n

        # 剪枝该层卷积的输出通道 (Conv2d is named .conv inside the standard block)
        target_module.conv.weight.data = target_module.conv.weight.data[keep_idxs]
        target_module.conv.out_channels = n

        ## c. 利用index对conv1进行剪枝 (Bias)=========================
        if target_module.conv.bias is not None:
            target_module.conv.bias.data = target_module.conv.bias.data[keep_idxs]

        ## d. 利用index对conv2 (下一层) 进行剪枝 (Input Channels)=========================
        if not isinstance(conv2, list):
            conv2 = [conv2]

        for item in conv2:
            if item is None: continue

            # 寻找下一层的实际 nn.Conv2d 对象
            next_conv = item

            # 如果是 Ultralytics 的 Conv 模块，真正的卷积层在 .conv 属性中
            if hasattr(item, 'conv') and isinstance(item.conv, nn.Conv2d):
                next_conv = item.conv
            # 兼容 RFAConv 作为下一层输入的情况（注意：这可能比较危险，因为 RFAConv 内部有分组卷积）
            elif hasattr(item, 'conv') and hasattr(item.conv, 'conv') and isinstance(item.conv.conv, nn.Conv2d):
                # RFAConv -> item.conv (Custom Conv) -> item.conv.conv (nn.Conv2d)
                # 注意：这里我们只剪枝了 RFAConv 最后输出部分的输入，但 RFAConv 前端的注意力生成部分
                # (get_weight, generate_feature) 的输入通道没有被剪枝，这会导致维度不匹配报错。
                # 建议：不要将 RFAConv 放在需要被剪枝连接的后半部分 (conv2)，
                # 或者在这里添加复杂的逻辑来处理 RFAConv 的输入结构。
                # 为了防止报错，这里暂时指向最后的卷积，但请确保 RFAConv 不是作为 conv2 传入的。
                next_conv = item.conv.conv
            elif isinstance(item, nn.Conv2d):
                next_conv = item

            # 执行剪枝：修改输入通道数和权重
            if isinstance(next_conv, nn.Conv2d):
                next_conv.in_channels = n
                next_conv.weight.data = next_conv.weight.data[:, keep_idxs]
            else:
                print(f"Warning: Could not prune input of {type(item)}, structure unknown.")

    def prune(self, m1, m2):
        if isinstance(m1, C2f):  # C2f as a top conv
            m1 = m1.cv2
        if not isinstance(m2, list):  # m2 is just one module
            m2 = [m2]
        for i, item in enumerate(m2):
            if isinstance(item, C2f) or isinstance(item, SPPF):
                m2[i] = item.cv1
        self.prune_conv(m1, m2)


def do_pruning(modelpath, savepath):
    pruning = PRUNE()

    ### 0. 加载模型
    yolo = YOLO(modelpath)
    pruning.get_threshold(yolo.model, 0.8)

    ### 1. 剪枝 c2f 中的 Bottleneck
    # 这里的 m.cv1 和 m.cv2 如果被你替换成了 RFAConv，现在新的 prune_conv 也能处理了
    for name, m in yolo.model.named_modules():
        if isinstance(m, Bottleneck):
            pruning.prune_conv(m.cv1, m.cv2)

    ### 2. 指定剪枝不同模块之间的卷积核
    seq = yolo.model.model
    # 请确保你在 yaml 中替换层数时，这些索引 i 依然对应的是 Conv-like 模块
    for i in [3, 5, 7, 8]:
        pruning.prune(seq[i], seq[i + 1])

    ### 3. 对检测头进行剪枝
    detect: Detect = seq[-1]
    last_inputs = [seq[19], seq[23], seq[27], seq[31]]
    colasts = [seq[20], seq[24], seq[28], None]

    for last_input, colast, cv2, cv3 in zip(last_inputs, colasts, detect.cv2, detect.cv3):
        pruning.prune(last_input, [colast, cv2[0], cv3[0]])
        pruning.prune(cv2[0], cv2[1])
        pruning.prune(cv2[1], cv2[2])
        pruning.prune(cv3[0], cv3[1])
        pruning.prune(cv3[1], cv3[2])

    ### 4. 模型梯度设置与保存
    for name, p in yolo.model.named_parameters():
        p.requires_grad = True

    yolo.val(workers=0, data='data.yml')
    torch.save(yolo.ckpt, savepath)
    yolo.model.pt_path = yolo.model.pt_path.replace("last.pt", os.path.basename(savepath))
    # 导出时可能会因为 RFAConv 的特殊算子导致问题，如果只是为了训练后微调，可以暂时注释掉 export
    # try:
    #     yolo.export(format="onnx")
    # except Exception as e:
    #     print(f"Export warning: {e}")

    ## 重新load模型，修改保存命名，用以比较剪枝前后的onnx的大小
    # yolo = YOLO(modelpath)
    # yolo.export(format="onnx")


if __name__ == "__main__":
    # 确保路径正确
    modelpath = "runs/detect1/14_Constraint/weights/last.pt"
    savepath = "runs/detect1/14_Constraint/weights/last_prune.pt"
    if os.path.exists(modelpath):
        do_pruning(modelpath, savepath)
    else:
        print(f"Error: model file not found at {modelpath}")
