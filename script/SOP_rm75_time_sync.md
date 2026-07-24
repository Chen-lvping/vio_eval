# RM75 时间同步 SOP

目标：把远端设备的系统时钟和 PTP PHC 对齐到本机时钟。

适用默认环境：

- 远端 IP：`192.168.2.240`
- SSH 用户：`ubuntu`
- SSH 密码：`ubuntu`
- 远端当前业务口：`eth1`
- 本机有线口：`enp3s0`

## 1. 直接同步

设备已经能从本机直连时，直接运行：

```bash
python3 script/sync_remote_ptp_to_local.py
```

脚本会自动做这些事：

- SSH 登录远端
- 自动识别远端接口和对应 PHC
- 多回合把远端系统时钟对齐到本机
- 把远端 PHC 跟到 `CLOCK_REALTIME`

## 2. 本机网段不通时

如果 `192.168.2.240` 当前不可达，让脚本先给本机有线口补一个临时地址：

```bash
python3 script/sync_remote_ptp_to_local.py \
  --local-iface enp3s0 \
  --local-ip-cidr 192.168.2.100/24
```

## 3. 同步后验证

用 SDK 视角验证控制器时间是否已经和本机一致：

```bash
python3 script/check_time_sync.py --ip 192.168.2.240 --count 3 --interval 0.3
```

或直接让同步脚本在结尾自动验证：

```bash
python3 script/sync_remote_ptp_to_local.py --verify-sdk-time
```

## 4. 期望结果

- `check_time_sync.py` 显示 `✓ SYNCED`
- `sync_remote_ptp_to_local.py` 最后输出的 PHC 对 `CLOCK_REALTIME` 偏差是微秒到纳秒级
- 多回合后 `after` 的 mean offset 通常会收敛到 `10 ms` 以内

## 5. 常用补充参数

```bash
# 远端不是默认 IP
python3 script/sync_remote_ptp_to_local.py --remote-ip <ip>

# 远端账号或密码变了
python3 script/sync_remote_ptp_to_local.py \
  --remote-user <user> \
  --remote-password <password>

# 只看探测结果，不真正设时
python3 script/sync_remote_ptp_to_local.py --dry-run
```

## 6. 失败时先看这三项

- 本机能否 `ping 192.168.2.240`
- 本机 `enp3s0` 是否接在设备所在网段
- 远端 SSH 密码和 `sudo` 密码是否可用
