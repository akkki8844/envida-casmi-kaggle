"""Submit a committed Kaggle notebook version to the competition (code competition), printing any error body.

    python scripts/submit_kernel.py govindmulchandani/<slug> <version> "<message>"
"""
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "enveda-CASMI26-molecule-id-mass-spectra"


def main():
    kernel, version, message = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    api = KaggleApi()
    api.authenticate()
    try:
        r = api.competition_submit_code(file_name="submission.csv", message=message, competition=COMP, kernel=kernel,
                                        kernel_version=version)
        print("ok", r)
    except Exception as e:  # show the server's reason
        print(type(e).__name__, e)
        resp = getattr(e, "response", None)
        if resp is not None:
            print(getattr(resp, "text", resp))
        return 1
    for s in api.competition_submissions(COMP)[:3]:
        print(s.ref, s.date, s.status, s.description[:80])
    return 0


if __name__ == "__main__":
    sys.exit(main())
