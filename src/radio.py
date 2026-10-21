"""LoRa airtime of each transmission strategy, from the Semtech SX1276 datasheet formula (section 4.1.1.7).

Radio settings are the usual LoRaWAN EU868 ones: 125 kHz, coding rate 4/5, 8-symbol preamble,
explicit header, CRC on, low data rate optimisation at SF11 and SF12.
"""
import math

BW = 125_000
PREAMBLE = 8
LORAWAN_OVERHEAD = 13  # MHDR 1 + FHDR 7 + FPort 1 + MIC 4 bytes around the application payload
MAX_PAYLOAD = {7: 222, 8: 222, 9: 115, 10: 51, 11: 51, 12: 51}  # EU868 application payload, LoRaWAN Regional Parameters
DUTY_CYCLE = 0.01  # EU868 sub-band limit: 1 % of the time per device, 864 s a day


def airtime_s(app_bytes: int, sf: int) -> float:
    """Time on air of one uplink carrying `app_bytes` of application payload."""
    return phy_airtime_s(app_bytes + LORAWAN_OVERHEAD, sf)


def phy_airtime_s(pl: int, sf: int) -> float:
    """Time on air of a LoRa packet with a `pl`-byte PHY payload."""
    t_sym = 2 ** sf / BW
    de = 1 if sf >= 11 else 0
    n = 8 + max(math.ceil((8 * pl - 4 * sf + 28 + 16) / (4 * (sf - 2 * de))) * 5, 0)  # CR 4/5 -> 5, CRC on, explicit header
    return (PREAMBLE + 4.25) * t_sym + n * t_sym


def frames(app_bytes: int, sf: int) -> int:
    return math.ceil(app_bytes / MAX_PAYLOAD[sf])


def message_airtime_s(app_bytes: int, sf: int) -> float:
    """Airtime of a message split into as many full uplinks as the data rate allows."""
    n = frames(app_bytes, sf)
    last = app_bytes - (n - 1) * MAX_PAYLOAD[sf]
    return (n - 1) * airtime_s(MAX_PAYLOAD[sf], sf) + airtime_s(last, sf)
