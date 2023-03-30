import os
import datetime

import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import altair as alt
from sklearn.linear_model import LinearRegression

# Функция для загрузки данных
@st.cache_resource
def load_data():
    path = "Dataset_master.csv.gz"
    if not os.path.isfile(path):
        path = f"https://github.com/SanyaZmei133/Waste_Visualisation/raw/master/{path}"

    data = pd.read_csv(
        path,
        #names=[
         #   "date/time",
         #   "PM10"
         #  "lat",
         #  "lon",
        #],
        #usecols=[1, 2, 3, 4],
        #skiprows=1,
        #index_col=0,
        #parse_dates=[
        #    "date/time"
        #],
    )
    return data

# Функция для фильтрации данных по дате и времени
@st.cache_data
def filterdata(df,date_selected, hour_selected):
    df["date/time"]=pd.to_datetime(df["date/time"])
    #df = df[df['date/time'].dt.date == date_selected]
    df = df[df["date/time"].dt.hour == hour_selected]
    return df

# Функция для вычисления центра карты
@st.cache_data
def mpoint(lat, lon):
    return (np.average(lat), np.average(lon))

# Функция для подготовки данных графика
@st.cache_data
def histdata(df):
    df["date/time"] = pd.to_datetime(data["date/time"])
    df["date/time"] = df["date/time"].dt.date
    df = df.drop('lat', axis=1)
    df = df.drop('lon', axis=1)
    PM10 = {
        "PM10": ['mean']
    }
    bar_data = data.groupby(['date/time']).agg(PM10).reset_index()


    return bar_data

# Функция для машинного обучения
@st.cache_data
def prediction(df):
    df['date/time'] = pd.to_datetime(data['date/time'])
    df['date_delta'] = (df['date/time'] - df['date/time'].min()) / np.timedelta64(1, 'D')

    df = df.drop('lat', axis=1)
    df = df.drop('lon', axis=1)
    df = df.drop('date/time', axis=1)

    X = df.drop('PM10', axis=1)
    y = df['PM10']

    X_train = X[:-72]
    X_test = X[-72:]
    y_train = y[:-72]
    y_test = y[-72:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return pd.DataFrame({"PM10":predictions, "hours":range(72)})

# Функция для карты
def map(data, lat, lon, zoom):
    tooltip = {
        "html": " PM10: <b>{PM10}</b> мг/м3",
        "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
    }
    st.write(
        pdk.Deck(
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state = {
                "latitude": lat,
                "longitude": lon,
                "zoom": zoom,
                "pitch": 50,
            },
            layers = [
                pdk.Layer(
                    'Hexagonlayer',
                    data=data,
                    get_position='[lon, lat]',
                    get_color='[200, 30, 0, 160]',
                    auto_highlight=True,
                    radius=100,
                    get_elevation='PM10',
                    elevation_scale=4,
                    elevation_range=[0, 0.1],
                    pickable=True,
                    extruded=True,
                    coverage=1,
                ),
                pdk.Layer(
                    'ColumnLayer',
                    data=data,
                    get_position='[lon, lat]',
                    get_evaluation='PM10',
                    evaluation_scale=1000,
                    radius=50,
                    get_fill_color=['PM10*1000','PM10*1000', 'RM10*100', 20],
                    pickable=True,
                    auto_highlight=True,
                ),
                pdk.Layer(
                    'ScatterplotLayer',
                    data=data,
                    get_position='[lon, lat]',
                    get_color='[200, 30, 0, 160]',
                    get_radius=100,
                ),
            ],
        )
    )


#Обновлени при перемещении слайдера
def update_query_params():
    hour_selected = st.session_state["tracked_hour"]
    date_selected = st.session_state["tracked_date"]
    st.experimental_set_query_params(tracked_date=date_selected, tracked_hour = hour_selected)

def update_query_params2():
    date_selected = st.session_state["tracked_date"]
    st.experimental_set_query_params(tracked_date=date_selected)

# Создание страницы приожения
st.set_page_config(layout = "wide", page_title="Данные о состоянии воздуха", page_icon="cloud:")

data = load_data()

row1_1, row1_2 = st.columns((2, 2))
with row1_1:
    st.title("Наблюдения за качеством воздуха в городе Пермь")
    hour_selected = st.slider(
        "Выберите время для наблюдений", 0, 23, value=0, key="tracked_hour", on_change=update_query_params
    )
with row1_2:
    st.write(
        """
            ##
            Мониторинг качетсва атмосферного воздуха с разных постов сбора образцов в городе Пермь по времени.     
            Выберите интересующую вас дату и перемещайте слайдер, чтобы просмотреть данные на конкретный момент времени и исследовать данные.
            """
    )
    date_selected = st.date_input(
        "Дата",
        value=datetime.date(2021, 12, 31),
        min_value=datetime.date(2021, 12, 31),
        max_value=datetime.date(2022, 1, 25),
        key="tracked_date", on_change=update_query_params
    )

midpoint = mpoint(data["lat"], data["lon"])
st.write(
f"""**Все наблюдения с {hour_selected}:00 до {(hour_selected + 1) % 24}:00**"""
)
map(filterdata(data,date_selected, hour_selected), midpoint[0], midpoint[1], 11)

st.sidebar.write(
    """
        ##
        # Дополнительные инструменты анализа данных          
        Выберите нужный инструмент.
        """
)

chart_data = histdata(data)

graph = st.sidebar.checkbox("Показать график")

if graph:
    st.write(
        f"""**Среднесуточные значения PM10 за весь период наблюдений**"""
    )

    st.altair_chart(
        alt.Chart(chart_data)
        .mark_area(
            interpolate="step-after",
        )
        .encode(
            x=alt.X("date/time:T", title = 'Date'), #scale=alt.Scale(nice=False)),
            y=alt.Y("PM10:Q"),
            tooltip=["date/time", "PM10"],
        )
        .configure_mark(opacity=0.2, color="red"),
        use_container_width=True,
    )
row2_1,row2_2 = st.columns((2, 2))

with row2_1:
    def highlight_PM10(val):
        color = 'yellow' if val > 0.05 else ''
        return f'background-color: {color}'

    st.dataframe(chart_data.style.applymap(highlight_PM10, subset = ['PM10']), use_container_width=True)
with row2_2:
    st.write(
        """
            ##
            Допустимое среднесуточное содержание частиц PM10 = 0,05 мкг/м3    
            В таблице видны дни когда значение превышает норму.
            """
    )

prediction_data = prediction(data)

prediction = st.sidebar.checkbox("Показать прогноз")

if prediction:
    st.write(
        """**Построение прогноза на следующие 72 часа**"""
    )
    st.altair_chart(
        alt.Chart(prediction_data)
        .mark_line(point={
            "filled": False,
            "fill": "white"
        })
        .encode(
            x=alt.X('hours:Q', title = "Hours",),
            y=alt.Y('PM10', title = "PM10", scale=alt.Scale(domain=[0.075,0.085])),
            tooltip=['PM10','hours'],
        )
        .configure_mark(color="red"),
        use_container_width=True,
    )