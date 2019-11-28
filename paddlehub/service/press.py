# -*- coding:utf-8 -*-
from time import time
import sys
import multiprocessing
from bert_service_test import BertService

#t_num = sys.argv[1]
batch_size = 1
data_list = []
with open("./check/data-c.txt") as f:
    for line in f.readlines():
        data_list.append([line.strip()])
print(len(data_list))
sys.stdout.flush()

start = time()


def process(batch_size,
            turn,
            check=False,
            profile=True,
            max_seq_len=128,
            model_name='bert_uncased_L-12_H-768_A-12',
            server_list=['127.0.0.1']):
    if check:
        check_list = []
        with open("./check/check-cased.txt") as f:
            for line in f.readlines():
                line = line.strip().split(" ")[5:]
                check_list.append(line)
        print(len(check_list[0]))

    bc = BertService(profile=profile,
                     emb_size=768,
                     model_name=model_name,
                     do_lower_case=True,
                     max_seq_len=max_seq_len)
    bc.connect_all_server(server_list)
    p_start = time()
    total_time = 0
    op_time = 0
    infer_time = 0
    request_time = 0
    max_diff = 0
    copy_time = 0
    connect_time = 0
    net_time = 0
    json_time = 0
    load_time = 0
    read_cost = 0
    request_cost = 0
    if profile:
        for i in range(turn):
            re_time = bc.encode(data_list[i:i + batch_size])
            total_time += re_time[0]
            request_time += re_time[1]
            op_time += re_time[2]
            infer_time += re_time[3]
            copy_time += re_time[4]
            #
            connect_time += re_time[5]
            net_time += re_time[6]
            json_time += re_time[7]
            load_time += re_time[8]
            read_cost += re_time[9]
            request_cost += re_time[10]
            #
            if i == 0:
                print("first time cost:" + str(total_time))
        p_end = time()

        print("batch_size:" + str(batch_size) + " " + str(i + 1) +
              " query cost " + str(p_end - p_start) + "s" + " total_time: " +
              str(total_time) + "s" + " request_time: " + str(request_time) +
              "s" + " copy_time: " + str(copy_time) + "s" + " op_time: " +
              str(op_time / 1000) + "s" + " infer_time: " +
              str(infer_time / 1000) + "s" + " connect_time: " +
              str(connect_time) + "s" + " net_cost: " + str(net_time) + "s" +
              " json cost: " + str(json_time) + "s load cost: " +
              str(load_time) + "s" + " read cost: " + str(read_cost) + "s " +
              "request time: " + str(request_cost) + "s")
    elif check:
        for i in range(turn):
            result = bc.encode(data_list[i:i + batch_size])
            for k in range(batch_size):
                for j in range(768):
                    diff = float(result[k][j]) - float(check_list[i + k][j])
                    if abs(diff) > max_diff:
                        max_diff = abs(diff)
                    if abs(diff) > 0.01:
                        print([result[k][j], check_list[i + k][j]])
                        print(data_list[i])
                        return -1

        print(max_diff)
    sys.stdout.flush()


for i in [256]:
    process(batch_size=i,
            turn=100,
            profile=True,
            check=False,
            max_seq_len=128,
            model_name='bert_chinese_L-12_H-768_A-12',
            server_list=['127.0.0.1:8010'])
