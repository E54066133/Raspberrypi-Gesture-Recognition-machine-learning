# Raspberrypi_Gesture-Recognition
use machine learning to  recognize gesture on raspberrypi 

## 說明
利用 tensorflow lite 訓練手部辨識模型  
分辨 "剪刀"、"石頭"、"布" 之手勢  
再將訓練模型匯入 Raspberry pi  
透過相機模組進行即時手勢判斷  
 

##  測試結果
布 =>  紅燈 (上面那顆)  
剪刀 => 黃燈 (下面那顆)  
石頭 => 綠燈 (中間那顆)  

demo 影片 => https://www.youtube.com/watch?v=6-rEq0tAKrw






##  結果分析
1. 在光源不足的地方辨識度會下降  
2. 背景複雜辨識度下降  
3. 石頭辨識度不高(需要特定角度)  
