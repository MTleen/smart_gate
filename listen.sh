#!/bin/sh
workdir=$(cd $(dirname $0); pwd)
ps -fe | grep -v grep | grep /home/mathripper/anaconda3/envs/torch/bin/python
if [ $? -ne 0 ]
then
{
	date > $workdir/log/listen.log
	echo "detecting process stoped, restart" >> $workdir/log/listen.log
	nohup /home/mathripper/anaconda3/envs/torch/bin/python -u /home/mathripper/Docker/smart_gate/detect.py --classes 0 2 --source rtsp://admin:HikLZDADB@192.168.1.27:554/h264/ch1/main > /dev/null 2>&1 &
}
else
{
	date > $workdir/log/listen.log
	echo "detecting is running" >> $workdir/log/listen.log
}
fi
