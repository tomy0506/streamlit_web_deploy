import streamlit as st
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
#from sklearn.model_selection import KFold
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['font.size'] = 15 
matplotlib.rcParams['axes.unicode_minus'] = False
st.set_option('deprecation.showPyplotGlobalUse', False)
# 앱의 타이틀 설정
st.title("기상 정보를 바탕으로 한 전력 수요 예측")
#train_data (전력 수요)
supply = pd.read_csv("C:/Users/one/Desktop/2차 인사이콘/전기.csv", encoding='ANSI')
supply["기준일시"] = pd.to_datetime(supply["기준일시"])
supply = supply[['기준일시', '현재수요(MW)']]
#test_data(전력 수요)
supply_want = pd.read_csv("C:/Users/one/Desktop/2차 인사이콘/elec.csv")
supply_want = supply_want.rename(columns ={"현재부하(MW)":"현재수요(MW)"})
supply_want = supply_want.loc[:, ["일시", "현재수요(MW)"]]
supply_want["일시"] = pd.to_datetime(supply_want["일시"])
for i in range(len(supply_want)):
    supply_want.loc[i, "현재수요(MW)"] = int(supply_want["현재수요(MW)"].str.split(",")[i][0]+supply_want["현재수요(MW)"].str.split(",")[i][1])
supply_want["현재수요(MW)"] = supply_want["현재수요(MW)"].astype(int)
#train_data(날씨)
weather = pd.read_csv("C:/Users/one/Desktop/2차 인사이콘/서울기상자료_5분단위.csv")
weather["일시"] = pd.to_datetime(weather["일시"])

#test_data(날씨)
weather_want = pd.read_csv("C:/Users/one/Desktop/2차 인사이콘/기상.csv",encoding = "ANSI")
weather_want["일시"] = pd.to_datetime(weather_want["일시"])
weather_want = weather_want[weather_want["일시"].between("2023-03-13", "2023-03-19 23:59")]
weather_want['요일'] = weather_want["일시"].apply(lambda x: x.weekday())
weather_want['월'] = weather_want["일시"].apply(lambda x: x.month)
weather_want.index = weather_want['일시']
weather_want.drop(columns=['일시'], inplace=True)
weather_want = weather_want.resample('5T').mean()

#날씨 전력 데이터 하나로 합치기(train)
df = pd.merge(weather, supply, left_on='일시', right_on='기준일시')
df.drop(columns=['기준일시'], inplace=True)

##날씨 전력 데이터 하나로 합치기(test)
df_want = pd.merge(weather_want, supply_want, left_on='일시', right_on='일시')

#요일, 월, 주말 변수 추가
from datetime import datetime,timedelta # datetime 라이브러리 임포트
import calendar
df['요일'] = df["일시"].apply(lambda x: x.weekday())
df['월'] = df["일시"].apply(lambda x: x.month)

df["주말"] = 0
for i in range(len(df["요일"])):
    if (df["요일"][i]==5 or df["요일"][i]==6):
        df.iloc[i, -1] = 1

df_want["주말"] = 0
for i in range(len(df_want["요일"])):
    if (df_want["요일"][i]==5 or df_want["요일"][i]==6):
        df_want.iloc[i, -1] = 1
#전처리 
df2 = df.copy()
df2.index = df2['일시']
df2.drop(columns=['일시'], inplace=True)
tmp= df2[['기온(°C)', '누적강수량(mm)', '현지기압(hPa)', '해면기압(hPa)', '습도(%)', '일사(MJ/m^2)', '일조(Sec)', '현재수요(MW)']].interpolate(method = "quadratic")
temp = df2[['풍향(deg)','풍속(m/s)']].interpolate(method = "nearest")
te = df2[['요일','월','주말']]
df3 = pd.concat([tmp,temp,te],axis = 1)

df4 = df_want.copy()
df4.index = df4['일시']
df4.drop(columns=['일시'], inplace=True)
tmp= df4[['기온(°C)', '누적강수량(mm)', '현지기압(hPa)', '해면기압(hPa)', '습도(%)', '일사(MJ/m^2)', '일조(Sec)', '현재수요(MW)']].interpolate(method = "quadratic")
te = df4[['요일','월','주말']]
temp = df4[['풍향(deg)','풍속(m/s)']].interpolate(method = "nearest")
df5 = pd.concat([tmp,temp,te],axis = 1)
#정상성 검정
df3["기온"] = df3["기온(°C)"].diff(1)
df3["기온"] = df3["기온"].fillna(0)
df5["기온"] = df5["기온(°C)"].diff(1)
df5["기온"] = df5["기온"].fillna(0)

df3["현지기압"] = np.log(df3["현지기압(hPa)"])
df3["현지기압"] = df3["현지기압"].diff(1)
df3["현지기압"] = df3["현지기압"].fillna(0)

df5["현지기압"] = np.log(df5["현지기압(hPa)"])
df5["현지기압"] = df5["현지기압"].diff(1)
df5["현지기압"] = df5["현지기압"].fillna(0)

df3["해면기압"] = np.log(df3["해면기압(hPa)"])
df3["해면기압"] = df3["해면기압"].diff(1)
df3["해면기압"] = df3["해면기압"].fillna(0)

df5["해면기압"] = np.log(df5["해면기압(hPa)"])
df5["해면기압"] = df5["해면기압"].diff(1)
df5["해면기압"] = df5["해면기압"].fillna(0)

df3["풍향"] = df3["풍향(deg)"].diff(1)
df3["풍향"] = df3["풍향"].fillna(200)

df5["풍향"] = df5["풍향(deg)"].diff(1)
df5["풍향"] = df5["풍향"].fillna(200)

df3["풍속"] = df3["풍속(m/s)"].diff(1)
df3["풍속"] = df3["풍속"].fillna(2)

df5["풍속"] = df5["풍속(m/s)"].diff(1)
df5["풍속"] = df5["풍속"].fillna(2)

df3["일사"] = df3['일사(MJ/m^2)'].diff(1)
df3["일사"] = df3["일사"].fillna(10)

df5["일사"] = df5['일사(MJ/m^2)'].diff(1)
df5["일사"] = df5["일사"].fillna(10)
#피쳐 선택
df6 = df3[['기온(°C)', '누적강수량(mm)', '현지기압(hPa)', '해면기압(hPa)', '풍향(deg)', '풍속(m/s)', '일사(MJ/m^2)', '일조(Sec)', '현재수요(MW)', '요일', '월',]]
df7 = df5[['기온(°C)', '누적강수량(mm)', '현지기압(hPa)', '해면기압(hPa)', '풍향(deg)', '풍속(m/s)', '일사(MJ/m^2)', '일조(Sec)', '현재수요(MW)', '요일', '월',]]

#테스트, 훈련 데이터 분리
train_data = df6
test_data = df7
X_train=train_data.drop('현재수요(MW)', axis=1)
y_train=train_data[['현재수요(MW)']]
X_test=test_data.drop('현재수요(MW)', axis=1)
y_test=test_data[['현재수요(MW)']]
date_string = st.slider("날짜를 선택하세요",
              min_value = datetime(2023,3,13,0,0,0), 
              max_value = datetime(2023,3,19,0,0,0),
              step = timedelta(minutes = 5),
              format = "MM-DD-YY - HH:mm"
)

X_test = X_test.loc[date_string:date_string+timedelta(hours=6)]
y_test = y_test.loc[date_string:date_string+timedelta(hours=6)]
#모델링
import catboost as cb
model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=4)
model.fit(X_train, y_train)
y_pred = y_test.copy()
y_pred['현재수요(MW)'] = 0
y_pred['현재수요(MW)'] = model.predict(X_test)
button = st.button("Mape 수치를 확인해보세요")
if button:
    st.markdown(mean_absolute_percentage_error(y_test, y_pred))

# 시각화
legend = ["observations", "median prediction"]
fig, ax = plt.subplots(1, 1, figsize=(20, 8))
y_test.plot(ax=ax)
y_pred.plot(ax=ax)
plt.grid(which="both")
plt.legend(legend, loc="upper left")
plt.xlabel("Timestamp")
plt.xticks(rotation=10)
plt.ylabel("Electricity Consumption")
plt.title("CatBoostRegressor Consumption Forecast")
st.pyplot(fig)