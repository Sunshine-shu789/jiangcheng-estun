# CEPB 数据集文件清单（http://cepbbenchmark.eu/ 的 Dataset 页）

来源：CEPB = Cluttered Environment Picking Benchmark，文件托管在圣安娜高等研究院
（Scuola Superiore Sant'Anna）的 SharePoint，匿名可下。任务书里点名了这个数据集
（"识别和抓取的物体可以参考 ICRA2025 … Picking in Cluster 赛道中提供的 40 种物体，
有关抓取对象数据集可以参考网址 http://cepbbenchmark.eu/ 提供的可下载的数据集"）。

联系：Salvatore D'Avella <salvatore.davella@santannapisa.it>

## 已下载

实际存放位置：数据盘 `/media/sunshine/新加卷/`。

| 文件 | 大小 | 校验 | 内容 |
| --- | --- | --- | --- |
| `models.zip` | 134,226,217 B（128 MiB） | sha256 `80305efccfe1efcbb702268e72b1f15b2cb0c07cbbda1db6ffff512c55b99355` | 39 个文件夹的网格（obj/dae + mtl + 贴图），覆盖赛道一 40 件物体 |
| `cameras_parameters.yaml` | 2,542 B | 内容已核对 | 仿真相机内参 K（f=3200, cx=1024, cy=768）、三个相机位姿、深度换算公式（near 900 mm / far 1400 mm） |
| `subset1_rand.zip` | 54,462,425,500 B | 与官方清单字节数一致，抽检 CRC 通过 | 5000 个场景 × 3 相机（left/middle/right），RGB + depth + normals + segmentation + GT yaml，共 105,000 个文件 |

`models.zip` 直链：

    https://alumnisssup-my.sharepoint.com/:u:/g/personal/salvatore_davella_santannapisa_it/IQAmKYbE4_OCTLRYRuEXIabmAQ0puyfcDih4-miKvTPLxfg?e=xsE2yH&download=1

## 未下载（感知训练数据，做视觉时才需要）

下面每个链接都要带 cookie 才不是 403：先用同一个 cookie jar（`curl -c jar -b jar`）
发一次 HEAD 或 GET 建立会话，再下正文。下载用 `-L` 跟随跳转，服务器支持
`accept-ranges: bytes` —— **单连接只有 30~60 KB/s，开 6 路 Range 并行能到 ~480 KB/s**。

| 文件 | 大小 |
| --- | --- |
| `subset3_rand.zip` | 40,596,865,472 B |
| `subset4_rand.zip` | 45,049,869,037 B |
| `subset2_rand.zip` | 46,878,110,567 B |
| `subset3_clearbox.zip` | 48,340,070,964 B |
| `subset4_clearbox.zip` | 49,392,504,722 B |
| `subset2_clearbox.zip` | 51,684,591,660 B |
| `subset1_clearbox.zip` | 60,795,710,444 B |
| `full_rand.zip` | 101,335,084,929 B |
| `full_clearbox.zip` | 117,877,386,565 B |

剩余合计约 343 GB（不含 `full_*`）。`subset1~4` 是四个物品子集，`rand` = 随机杂乱摆放，
`clearbox` = 按比赛规程装进 IKEA SAMLA 透明箱（对应任务 B），`full_*` 是全集打包，
等价于前面几个的并集，不用单独下。每个 zip 里是 RGB-D 图像 +
`GT_left_camera_*.yaml` 之类的真值标注。
