"""Check that the training resource guard fails closed."""

from train_recurrent_cnn import ResourceGuardCallback


def main() -> None:
    guard = ResourceGuardCallback(check_every=1)
    guard.n_calls = 1
    assert guard._on_step() is True

    limited = ResourceGuardCallback(max_rss_mb=1, check_every=1)
    limited.n_calls = 1
    assert limited._on_step() is False
    print("Resource guard check passed")


if __name__ == "__main__":
    main()
