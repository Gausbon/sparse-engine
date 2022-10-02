import json

def params_parser(path):
    layer_list = {}
    with open(path) as rfile:
        params = json.load(rfile)
    for key, value in params.items():
        sum = 0
        if (type(value) == dict):
            for size in value.values():
                sum += size
            key_head = key.split('_')[0]
            if key_head in layer_list.keys():
                layer_list[key_head] += sum
            else:
                layer_list[key_head] = sum

    sum = 0
    for key, value in layer_list.items():
        print('key: ' + key)
        sum += value
        print('size: ' + str(sum))

if __name__ == '__main__':
    params_parser('../TinySPOS/logdir/tinyformer_dmtp_toy_dmtdmtp_config__block_3_quantization_tq/params.json')