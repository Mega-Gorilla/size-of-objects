#インポート
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# VideoCaptureのインスタンスを作成する。
# 引数でカメラを選べれる。
cap = cv2.VideoCapture(1)

while True:
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()

    ## 加工なし画像を表示する
    cv2.imshow('Original', frame)

    ## 輪郭形成　フィルタ適応
    image = frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #白黒処理
    #cv2.imshow('Gray', gray) #画面表示(デバック時)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)   #ブラー処理
    #cv2.imshow('Gray+Blur', gray) #画面表示(デバック時)
    edged = cv2.Canny(gray, 50, 100) #エッジ検出
    #cv2.imshow('edged', edged) #画面表示(デバック時)
    edged = cv2.dilate(edged, None, iterations=1) #モルフォロジー変換 - 膨張
    #cv2.imshow('edged(Dilation)', edged) #画面表示(デバック時)
    edged = cv2.erode(edged, None, iterations=1) #モルフォロジー変換 - 収縮
    #cv2.imshow('edged(Dilation+Erosion)', edged) #画面表示(デバック時)

    ##エッジ検出
    #cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE )
    #cv2.findContours(入力画像, 抽出モード, 近似手法)
    img = frame.copy() #プレビュー用カラー画像のコピー
    cnts = imutils.grab_contours(cnts)
    #img = cv2.drawContours(img, cnts, -1, (0,255,0), 3) #プレビュー版カラー画像にエッジ座標データを追加
    #cv2.imshow('edge2', img) #画面表示(デバック時)
    (cnts, _) = contours.sort_contours(cnts)
    #輪郭を上から下または下から上にソート数値の並び替え
    pixelsPerMetric = None
    
    #輪郭ごとの検出ループ
    for c in cnts:
        #輪郭が小さい場合無視 contourAreaは面積
        if cv2.contourArea(c) < 100:
            continue

        #回転を考慮した矩形輪郭の取得
        box = cv2.minAreaRect(c) #返戻値→(左上の点(x,y)，横と縦のサイズ(width, height)，回転角)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box) #不要なデータを消去
        box = np.array(box, dtype="int") #座標数値の配列化
        box = perspective.order_points(box) #輪郭内の点の並び替え、左上右上右下左下に
        cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2) #取得した座標情報の作画(緑)
    cv2.imshow('test', img) #画面表示(デバック時)

    # キー入力を1ms待って、k が27（ESC）だったらBreakする
    k = cv2.waitKey(1)
    if k == 27:
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()