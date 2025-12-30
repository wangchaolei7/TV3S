_base_ = ['./tv3s_b5_city_base.py']

data = dict(
    val=dict(corruption='fog'),
    test=dict(corruption='fog'))
