import random

from Decoder import *
from Encoder import *


def test_encode_decode():
    # test
    #create bytes io
    import io

    random_test_byte = random.choices(range(0xff), k=100)
    random_test_int = random.choices(range(0xffff), k=1000)
    random_test_long = random.choices(range(0xffffffff), k=1000)
    random_test_float = [1.1115,1.75,5.3,9.9,4.0,18.04511260986328]
        #[fl / random.choice(range(2,20)) for fl in random.choices(range(0xffff), k=1000)]
    random_test_string = ['Unique Ability / ユニークアビリティ','YuNi - ユニークアビリティ /','ßöäü+§!"§$%&/()=?`´^°*+~#-_.:,;<>|@€{[]}','','\n','\t']


    f = io.BytesIO()
    for b in [True,False]:
        reset(f)
        encode_bool(f, b)
        f.seek(0)
        j = decode_bool(f)
        if b != j:
            print(f"bool Failed: {b} != {j}")

    for i in random_test_byte:
        reset(f)
        encode_byte(f, i)
        f.seek(0)
        j = decode_byte(f)
        if i != j:
            print(f"byte Failed: {i} != {j}")

    for i in random_test_int:
        reset(f)
        encode_int(f, i)
        f.seek(0)
        j = decode_int(f)
        if i != j:
            print(f"int Failed: {i} != {j}")

    for i in random_test_long:
        reset(f)
        encode_long(f, i)
        f.seek(0)
        j = decode_long(f)
        if i != j:
            print(f"long Failed: {i} != {j}")

    for i in random_test_float:
        reset(f)
        encode_float(f, i)
        f.seek(0)
        j = decode_float(f)
        if i != j:
            print(f"float Failed: {i} != {j}")

    for idx,i in enumerate(random_test_string):
        #print(idx,i)
        reset(f)
        encode_string(f, i)
        #f.seek(0)
        #my_by= f.read();
        #with open('D:/_TMP/easy.test', 'wb') as fo:
        #    fo.write(my_by)
        f.seek(0)
        j = decode_string_maybe_utf16(f)
        if i != j:
            print(f"string Failed: {i} != {j}")

def reset(f):
    f.seek(0)
    f.truncate(0)
    return f

#main
if __name__ == '__main__':
    test_encode_decode()