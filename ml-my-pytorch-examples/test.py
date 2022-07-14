import logging

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

log = logging.getLogger(__name__)

if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        for i in tqdm(range(100)):
            if i % 25 == 0:
                log.warning(f"console logging redirected to `tqdm.write()` at {i}")

    # logging restored
