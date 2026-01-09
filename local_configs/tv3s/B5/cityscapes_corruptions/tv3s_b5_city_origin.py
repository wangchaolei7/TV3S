_base_ = ['./tv3s_b5_city_base.py']

data = dict(
    val=dict(img_dir='origin_leftImg8bit_sequence', corruption=None),
    test=dict(img_dir='origin_leftImg8bit_sequence', corruption=None),
)
