# coding:utf-8
import sys
import time
import numpy as np
import paddlehub as hub
import json
import random
from paddlehub.common.logger import logger

_ver = sys.version_info
is_py2 = (_ver[0] == 2)
is_py3 = (_ver[0] == 3)

if is_py2:
    import httplib
if is_py3:
    import http.client as httplib


class BertService():
    def __init__(self,
                 profile=False,
                 max_seq_len=128,
                 model_name="bert_uncased_L-12_H-768_A-12",
                 emb_size=768,
                 show_ids=False,
                 do_lower_case=True):
        self.reader_flag = False
        self.batch_size = 16
        self.embedding_size = emb_size
        self.max_seq_len = max_seq_len
        self.profile = profile
        self.model_name = model_name
        self.show_ids = show_ids
        self.do_lower_case = do_lower_case
        self.con_list = []
        self.con_index = 0
        self.load_balance = 'random'
        self.server_list = []

    def connect(self, ip='127.0.0.1', port=8010):
        self.server_list.append(ip + ':' + str(port))
        con = httplib.HTTPConnection(ip, port)
        self.con_list.append(con)

    def connect_all_server(self, server_list):
        for server_str in server_list:
            self.server_list.append(server_str)
            ip, port = server_str.split(':')
            port = int(port)
            self.con_list.append(httplib.HTTPConnection(ip, port))

    def data_convert(self, text):
        if self.reader_flag == False:
            module = hub.Module(name=self.model_name)
            self.reader = hub.reader.ClassifyReader(
                vocab_path=module.get_vocab_path(),
                dataset=None,
                max_seq_len=self.max_seq_len,
                do_lower_case=self.do_lower_case)
            self.reader_flag = True

        return self.reader.data_generator(batch_size=self.batch_size,
                                          phase='predict',
                                          data=text)

    def infer(self, request_msg):
        if self.load_balance == 'random_robin':
            try:
                cur_con = self.con_list[self.con_index]
                cur_con.request('POST', "/BertService/inference", request_msg,
                                {"Content-Type": "application/json"})
                response = cur_con.getresponse()
                response_msg = response.read()
                response_msg = json.loads(response_msg)
                self.con_index += 1
                self.con_index = self.con_index % len(self.con_list)
                return response_msg

            except BaseException as err:
                logger.warning("Infer Error with server {} : {}".format(
                    self.server_list[self.con_index], err))
                del self.con_list[self.con_index]
                del self.server_list[self.con_index]
                if len(self.con_list) == 0:
                    logger.error('All server failed, process will exit')
                    return 'fail'
                else:
                    self.con_index = 0
                    return 'retry'

        elif self.load_balance == 'random':
            try:
                self.con_index = random.randint(0, len(self.server_list) - 1)
                cur_con = httplib.HTTPConnection(
                    self.server_list[self.con_index])
                cur_con.request('POST', "/BertService/inference", request_msg,
                                {"Content-Type": "application/json"})
                response = cur_con.getresponse()
                response_msg = response.read()
                response_msg = json.loads(response_msg)

                return response_msg
            except BaseException as err:

                logger.warning("Infer Error with server {} : {}".format(
                    self.server_list[self.con_index], err))
                if len(self.server_list) == 0:
                    logger.error('All server failed, process will exit')
                    return 'fail'
                else:
                    self.con_index = 0
                    return 'retry'

    def encode(self, text):
        if type(text) != list:
            raise TypeError('Only support list')
        #start = time.time()
        self.batch_size = len(text)
        data_generator = self.data_convert(text)
        start = time.time()
        request_time = 0
        result = []
        for run_step, batch in enumerate(data_generator(), start=1):
            request = []
            copy_start = time.time()
            token_list = batch[0][0].reshape(-1).tolist()
            pos_list = batch[0][1].reshape(-1).tolist()
            sent_list = batch[0][2].reshape(-1).tolist()
            mask_list = batch[0][3].reshape(-1).tolist()
            for si in range(self.batch_size):
                instance_dict = {}
                instance_dict["token_ids"] = token_list[si *
                                                        self.max_seq_len:(si +
                                                                          1) *
                                                        self.max_seq_len]
                instance_dict["sentence_type_ids"] = sent_list[si *
                                                               self.max_seq_len:
                                                               (si + 1) *
                                                               self.max_seq_len]
                instance_dict["position_ids"] = pos_list[si *
                                                         self.max_seq_len:(si +
                                                                           1) *
                                                         self.max_seq_len]
                instance_dict["input_masks"] = mask_list[si *
                                                         self.max_seq_len:(si +
                                                                           1) *
                                                         self.max_seq_len]
                instance_dict["max_seq_len"] = self.max_seq_len
                instance_dict["emb_size"] = self.embedding_size
                request.append(instance_dict)
            copy_time = time.time() - copy_start
            #request
            request = {"instances": request}
            request_msg = json.dumps(request)
            if self.show_ids:
                logger.info(request_msg)
            request_start = time.time()
            response_msg = self.infer(request_msg)
            while type(response_msg) == str and response_msg == 'retry':
                logger.info('Try to connect another servers')
                response_msg = self.infer(request_msg)

            for msg in response_msg["instances"]:
                for sample in msg["instances"]:
                    result.append(sample["values"])

            #request end
            request_time += time.time() - request_start
        total_time = time.time() - start
        start = time.time()
        if self.profile:
            return [
                total_time, request_time, response_msg['op_time'],
                response_msg['infer_time'], copy_time
            ]
        else:
            return result

    def close(self):
        for con in self.con_list:
            con.close()


def test():

    bc = BertService(model_name='bert_uncased_L-12_H-768_A-12',
                     emb_size=768,
                     show_ids=False,
                     do_lower_case=True)
    bc.connect_all_server([
        '10.255.135.34:8010', '10.255.135.34:8011', '10.255.135.34:8012',
        '10.255.135.34:8013', '10.255.135.34:8014', '10.255.135.34:8015',
        '10.255.135.34:8016', '10.255.135.34:8017'
    ])

    for i in range(1000):
        text = [["As long as"] for i in range(256)]
        result = bc.encode(text)
    #print(result[0])
    bc.close()


if __name__ == '__main__':
    test()
