_base_ = ['./tv3s_b5_city_base.py']

data = dict(
    val=dict(corruption='snow'),
    test=dict(corruption='snow'))
