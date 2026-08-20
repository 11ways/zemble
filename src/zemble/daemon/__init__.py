"""The warm zemble daemon: one process per user holding indexes in RAM.

`zemble.daemon.client` is safe to import from short-lived processes; the server and
watcher halves pull in the index machinery and are imported only by the daemon itself.
"""
