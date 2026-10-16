import carla

for port in [2000, 2001, 2002, 3000, 4000, 8000]:
    try:
        client = carla.Client("127.0.0.1", port)
        client.set_timeout(2.0)
        server = client.get_server_version()
        world = client.get_world()
        print(f"OK port={port} server={server} map={world.get_map().name}")
    except Exception as exc:
        print(f"NO port={port} {type(exc).__name__}: {exc}")
