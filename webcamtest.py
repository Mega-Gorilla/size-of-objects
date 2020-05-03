#インポート
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# VideoCaptureのインスタンスを作成する。
# 引数でカメラを選べれる。
cap = cv2.VideoCapture(1)

#計算式- 中点計算用 線上の点
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


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
    overlap_tl=[]
    overlap_br=[]
    overlap_size = 10

    #輪郭ごとの検出ループ
    for c in cnts:
        #輪郭が小さい場合無視 contourAreaは面積
        if cv2.contourArea(c) < 500:
            continue

        #回転を考慮した矩形輪郭の取得
        box = cv2.minAreaRect(c) #返戻値→(左上の点(x,y)，横と縦のサイズ(width, height)，回転角)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box) #不要なデータを消去 座標データのみにする
        box = np.array(box, dtype="int") #座標数値の配列化
        box = perspective.order_points(box) #輪郭内の点の並び替え、左上右上右下左下に
        cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2) #取得した座標情報の作画(緑)
        # 点を作画
        for (x, y) in box:
            #座標、半径、色、太さ(塗りつぶし-1)
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        # 左上座標と右上座標の間の中間点を計算し，次に左下座標と右下座標の間の中間点を計算します．
        # tl -tltr - tr
        # /          /
        # tlbl      trbr
        # /          /
        # bl -blbr - br
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        overlap_tl.append([tl[0],tl[1]])
        overlap_br.append([br[0],br[1]])
        #左上点と右上点の間の中間点を計算し、その後に右上点と右下点の間の中間点を計算します。 左上点と右上点の間の中間点を計算し、その後に右上点と右下点の間の中間点を計算します。
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        # draw the midpoints on the image
        cv2.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        #内側に作画する場合は寸法を書かない

        # 中間点間のユークリッド距離を計算する
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY)) #四角高さ
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY)) #四角幅

        #ピクセル/mm　に値が入っていない時、値を入力
        if pixelsPerMetric is None:
            pixelsPerMetric = 17.5
            print(pixelsPerMetric)
        # compute the size of the object
        dimA = dA / pixelsPerMetric #dA 高さ dimA　mm高さ
        dimB = dB / pixelsPerMetric
        # draw the object sizes on the image
        cv2.putText(img, "{:.1f}mm".format(dimA),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
        cv2.putText(img, "{:.1f}mm".format(dimB),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
    

    cv2.imshow('Measure', img) #画面表示(デバック時)

    # キー入力を1ms待って、k が27（ESC）だったらBreakする
    k = cv2.waitKey(1)
    if k == 27:
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()