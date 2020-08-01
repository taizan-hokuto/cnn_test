from PIL import Image
from keras.models import load_model
import numpy as np
import cv2

# カメラから画像を取得して,リアルタイムに手書き数字を判別させる。
# 動画表示

def overlayImage(src, overlay, location):
    overlay_height, overlay_width = overlay.shape[:2]

    # 背景をPIL形式に変換
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    pil_src = Image.fromarray(src)
    pil_src = pil_src.convert('RGBA')

    # オーバーレイをPIL形式に変換
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
    pil_overlay = Image.fromarray(overlay)
    pil_overlay = pil_overlay.convert('RGBA')

    # 画像を合成
    pil_tmp = Image.new('RGBA', pil_src.size, (255, 255, 255, 0))
    pil_tmp.paste(pil_overlay, location, pil_overlay)
    result_image = Image.alpha_composite(pil_src, pil_tmp)

    # OpenCV形式に変換
    return cv2.cvtColor(np.asarray(result_image), cv2.COLOR_RGBA2BGRA)


cap = cv2.VideoCapture(0)

model = load_model("sign.h5")  # 学習済みモデルをロード


# frame = cv2.imread('a.png')

# 無限ループ
while(True):

    # 判定用データの初期化
    Xt = []
    Yt = []

    ret, frame = cap.read()

    # 画像のサイズを取得,表示。グレースケールの場合,shape[:2]
    h, w, _ = frame.shape[:3]
    # h = 28
    # w = 28

    # 画像の中心点を計算
    w_center = w//2
    h_center = h//2

    # 画像の真ん中に142×142サイズの四角を描く
    cv2.rectangle(frame, (w_center-71, h_center-71), (w_center+71, h_center+71),(255, 0, 0))

    # カメラ画像の整形
    im = frame[h_center-70:h_center+70, w_center-70:w_center+70]  # トリミング
    # im = frame[0:28, 0:28]  
    th = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # グレースケールに変換
    # _, th = cv2.threshold(th, 0, 255, cv2.THRESH_OTSU) # 2値化
    # th = cv2.bitwise_not(th) # 白黒反転

    # th = cv2.GaussianBlur(th,(9,9), 0) # ガウスブラーをかけて補間
    # th = cv2.GaussianBlur(im, (9, 9), 0)  # ガウスブラーをかけて補間
    #
    th = cv2.resize(th, (28, 28), cv2.INTER_CUBIC)  # 訓練データと同じサイズに整形 ## orig im
    th = th.reshape(28, 28, 1)

    Xt.append(th)
    Xt = np.array(Xt)/255

    result = model.predict(Xt, batch_size=1)  # 判定,ソート

    ch = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm',
          'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

    for i in range(len(ch)):
        r = round(result[0, i], 2)
        Yt.append([ch[i], r])
        Yt = sorted(Yt, key=lambda x: (x[1]))

    # 判定結果を上位3番目まで表示させる
    # print([Yt[23],Yt[22],Yt[21]])
    cv2.putText(frame, "1:"+str(Yt[23]), (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "2:"+str(Yt[22]), (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "3:"+str(Yt[21]), (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

    frame = overlayImage(frame, th, (96, 96))

    cv2.imshow("frame", frame)  # カメラ画像を表示

    k = cv2.waitKey(1) & 0xFF  # キーが押下されるのを待つ。1秒置き。64ビットマシンの場合,& 0xFFが必要
    prop_val = cv2.getWindowProperty(
        "frame", cv2.WND_PROP_ASPECT_RATIO)  # アスペクト比を取得

    if k == ord("q") or (prop_val < 0):  # 終了処理
        break



cap.release()  # カメラを解放
cv2.destroyAllWindows()  # ウィンドウを消す
