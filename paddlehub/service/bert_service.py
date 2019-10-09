# coding:utf-8
import sys
import time
import numpy as np
import paddlehub as hub
import json

_ver = sys.version_info
is_py2 = (_ver[0] == 2)
is_py3 = (_ver[0] == 3)

if is_py2:
    import httplib
if is_py3:
    import http.client as httplib


class BertService():
    def __init__(self):
        self.reader_flag = False

    def connect(self, ip, port):
        self.con = httplib.HTTPConnection(ip, port)

    def data_convert(self, text):
        if self.reader_flag == False:
            module = hub.Module(name="bert_uncased_L-24_H-1024_A-16")
            dataset = hub.dataset.ChnSentiCorp()
            self.reader = hub.reader.ClassifyReader(
                dataset=dataset,
                vocab_path=module.get_vocab_path(),
                max_seq_len=128)
            self.reader_flag = True

        return self.reader.data_generator(
            batch_size=1, phase='predict', data=text)

    def encode(self, text):
        if type(text) != list:
            raise TypeError('Only support list')
        start = time.time()
        data_generator = self.data_convert(text)
        result = []
        for run_step, batch in enumerate(data_generator(), start=0):
            request = []
            #start = time.time()
            for sample in batch:
                instance_dict = {}
                instance_dict["token_ids"] = sample[0].reshape(-1).tolist()
                instance_dict["sentence_type_ids"] = sample[2].reshape(
                    -1).tolist()
                instance_dict["position_ids"] = sample[1].reshape(-1).tolist()
                instance_dict["input_masks"] = sample[3].reshape(-1).tolist()
                instance_dict["max_seq_len"] = 128
                request.append(instance_dict)
            request = {"instances": request}
            request_msg = json.dumps(request)
            #start = time.time()
            try:
                self.con.request('POST', "/BertService/inference", request_msg,
                                 {"Content-Type": "application/json"})
                response = self.con.getresponse()
                response_msg = response.read()
                response_msg = json.loads(response_msg)
                for msg in response_msg["instances"]:
                    for sample in msg["instances"]:
                        result.append(sample["values"])

            except httplib.HTTPException as e:
                print(e.reason)

            print('cost :' + str(time.time() - start))
            start = time.time()
        return result

    def close(self):
        self.con.close()


if __name__ == '__main__':
    bc = BertService()
    bc.connect('127.0.0.1', 8010)
    result = bc.encode([
        [
            "As a woman you shouldn't complain about cleaning up your house. &amp; as a man you should always take the trash out..."
        ],
        ["hello"],
    ])
    print(result)
    bc.close()
