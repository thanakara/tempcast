__job_id: str | None = None


def set_job_id(job_id: str) -> None:
    global __job_id  # noqa: PLW0603
    __job_id = job_id


def get_job_id() -> str:
    if __job_id is None:
        raise RuntimeError("`job_id` Not Set. Was `@on_job_start` called?")
    return __job_id
