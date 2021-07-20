import pickle
import streamlit as st
import pandas as pd

PATH_MODEL = 'model/forest_model.pkl'


def load_model(path: str, mode: str = 'rb'):
    with open(path, mode=mode) as fio:
        return pickle.load(fio)


class Classifier:

    def __init__(self):
        self.forest = load_model(PATH_MODEL)

    def predict(self, data):
        prediction = self.forest.predict(data)
        return prediction


classifier = Classifier()

st.header("Will the next slot be successful in the first weeks?")
st.text("")
st.text("")
st.sidebar.markdown("## About project")
st.sidebar.markdown(""" This is an attempt to find a variant 
                        of the most successful features in the game.
                        But this is not a guide for future releases. 
                        Try it for free, but make your own decision, 
                        machine learning can help, but it not able 
                        to replace the human mind in business matters. """)

left_column, right_column = st.beta_columns(2)

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

multiplier = left_column.checkbox('multiplier')
scatter = left_column.checkbox('scatter')
grand = left_column.checkbox('grand')
free_spins = right_column.checkbox('free_spins')
pick_bonus_game = right_column.checkbox('pick_bonus_game')
respin_game = right_column.checkbox('respin_game')


data = pd.DataFrame({'multiplier': [multiplier] * 1,
                     'scatter': [scatter] * 1,
                     'grand': [grand] * 1,
                     'free_spins': [free_spins] * 1,
                     'pick_bonus_game': [pick_bonus_game] * 1,
                     'respin_game': [respin_game] * 1})
st.text("")
st.text("")
left_column, center, right_column = st.beta_columns(3)
if center.button('predict!'):
    st.text("")
    if data.values.sum() > 3:
        st.write("You have selected too many features!")
    else:
        if classifier.predict(data) == 1:
            st.write('Cool! 😃 Slot with these features should show good results!')
        else:
            st.write('Oh No! 😟 Probably, you need to consider other features...')
