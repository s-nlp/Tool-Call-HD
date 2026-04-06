# Startup
Перед стартом создать ключи в файле `api_keys.conf`, пример:
```bash
"Bearer sk-first-dsadasd" 1;
"Bearer sk-second-asdas" 1;
```


# команды Make
```bash
make up
make down
make restart
make restart-nginx

make logs
make logs-vllm
make logs-nginx

make status
make health

make add-key USER=virven
make list-keys
make remove-key USER=alice

make test API_KEY=sk-alice-...

make clean
make clean-all
```