# WSDream Dataset1

This directory is used for WS-DREAM Dataset1. 

## Files

```text
dataset/wsdream
├── README.md
├── rtMatrix
├── tpMatrix
├── userlist
└── wslist
```

`.txt` suffixes are also supported.

## Statistics

| Item | Value |
|---|---:|
| Users | 339 |
| Web services | 5,825 |
| QoS records | 1,974,675 |
| QoS metrics | response time, throughput |

## Format

`rtMatrix` and `tpMatrix` are user-service QoS matrices. Rows are users, columns are services, and values are observed QoS values.

`userlist` stores user metadata, including user ID, IP, region, AS, latitude, and longitude.

`wslist` stores service metadata, including service ID, WSDL, IP, region, AS, latitude, and longitude.


## Reference

Zibin Zheng, Yilei Zhang, and Michael R. Lyu, “Investigating QoS of Real-World Web Services,” IEEE Transactions on Services Computing, 2014.
