#!/bin/bash

kill $(ps aux | grep '[p]ython run_qa.py' | awk '{print $2}')
