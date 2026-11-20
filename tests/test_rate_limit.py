from knowledge.utils.rate_limit import RateLimiter


class Clock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_rpm_limit_waits_for_the_window():
    c = Clock()
    rl = RateLimiter(rpm_limit=2, tpm_limit=1000, window_seconds=60, clock=c.time, sleep=c.sleep)
    rl.acquire(1)
    rl.acquire(1)
    assert c.slept == []
    rl.acquire(1)
    assert c.slept and c.now >= 60


def test_tpm_limit_waits_for_the_window():
    c = Clock()
    rl = RateLimiter(rpm_limit=100, tpm_limit=100, window_seconds=60, clock=c.time, sleep=c.sleep)
    rl.acquire(80)
    rl.acquire(30)
    assert c.now >= 60


def test_oversized_request_goes_through_alone():
    c = Clock()
    rl = RateLimiter(rpm_limit=100, tpm_limit=100, window_seconds=60, clock=c.time, sleep=c.sleep)
    rl.acquire(500)
    assert c.slept == []
