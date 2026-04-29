# ToolACE Processing Report

## Что было изначально

В исходном `ToolACE` было `11,300` диалогов.

Из них:

- `8,052` содержат хотя бы один `tool call`
- `7,543` содержат **ровно один** распознанный `tool call` в исходной строке датасета
- `235` попали в structural `multihop`-подмножество, где есть как минимум два завершённых tool-use шага
- `3,522` строки не попали в обработку `singlehop/multihop`

Что это за `3,522` строки:

- диалоги без `tool call`
- диалоги с несколькими tool call, но без нашей structural multi-hop цепочки
- нестандартные или незавершённые случаи, которые не проходят наши строгие правила отбора

![ToolACE Processing Counts](/datasets/toolace_processing_report/toolace_processing_counts.png)

![ToolACE Selection Breakdown](/datasets/toolace_processing_report/toolace_selection_breakdown.png)

## Как был выделен `multihop`

Для `multihop` брались только диалоги, где есть минимум два завершённых шага вида:

`user -> assistant(tool call) -> tool -> assistant(answer)`

После structural-фильтрации получилось `235` строк. Затем был выполнен cleaning, и основной real multihop-набор составил `186` строк. Дополнительно после synthetic-этапа было найдено ещё `5` multi-hop диалогов, которые были возвращены в multihop-часть датасета.

- `datasets/multihop/toolace_multistep_without_followups.json` -> `235`
- `datasets/multihop/toolace_multistep_clean.json` -> `186`
- `datasets/multihop/multihop_synthetic_toolace.json` -> `5`
- итоговый `multihop total` -> `191`

![Multihop Refinement](/datasets/toolace_processing_report/toolace_multihop_refinement.png)

Распределение по числу завершённых tool-use шагов в `multihop`:

- в `235` строках: `227` диалогов имеют `2` завершённых шага, `8` диалогов имеют `3`
- в `186` clean-строках: `180` диалогов имеют `2` завершённых шага, `6` диалогов имеют `3`
- ещё `5` synthetic multihop-строк были возвращены в multihop-часть датасета

![Multihop Completed Turn Distribution](/datasets/toolace_processing_report/toolace_multihop_completed_turn_distribution.png)

## Как был выделен `singlehop`

Для `singlehop` брались все строки, где в исходном диалоге есть **ровно один** распознанный `tool call`.

Таких строк получилось `7,543`.

Но внутри этого подмножества обнаружилось, что:

- только `204` строк содержат полный завершённый шаг `user -> assistant(tool call) -> tool -> assistant(answer)`
- `7,339` строк заканчиваются прямо на `assistant(tool call)`

То есть в большинстве `singlehop`-диалогов последний шаг не завершён внутри самой строки датасета.

![Singlehop Flow](/datasets/toolace_processing_report/toolace_singlehop_flow.png)

## Синтетическая генерация `tool response` + `assistant`

Для `7,339` строк, которые заканчивались на `tool call`, было сделано синтетическое достраивание:

1. сгенерирован `tool_response`
2. затем на его основе сгенерирован финальный `assistant`
3. строки с нежелательными паттернами вроде `error` / `unavailable` были удалены
4. из результата были убраны synthetic multi-hop артефакты

Итоговое разбиение:

- `singlehop_real`: `204`
- `singlehop_synthetic`: `7,293`
- `multihop_synthetic`: `5`

Текущие файлы:

- real completed singlehop: `datasets/singlehop/singlehop_real_toolace.json`
- synthetic singlehop: `datasets/singlehop/singlehop_synthetic_toolace.json`
- synthetic multihop: `datasets/multihop/multihop_synthetic_toolace.json`

## Сравнение длины диалогов

Средняя длина диалогов в сообщениях:

- весь `ToolACE`: `2.446`
- строки с `>=1 tool call`: `2.585`
- исходный `singlehop` с ровно 1 tool call: `2.159`
- real completed `singlehop` (`204`): `7.696`
- synthetic final `singlehop` (`7293`): `4.001`
- structural `multihop` `235`: `10.689`
- final clean `multihop` `186`: `10.688`
- final total `multihop` `191`: `10.681`

Средняя длина в символах:

- весь `ToolACE`: `624.681`
- исходный `singlehop exact1`: `489.734`
- real completed `singlehop` (`204`): `2867.877`
- synthetic final `singlehop` (`7293`): `1549.138`
- structural `multihop` `235`: `3826.285`
- final clean `multihop` `186`: `3760.237`
- final total `multihop` `191`: `3770.99`

![Average Dialogue Messages](/datasets/toolace_processing_report/toolace_avg_dialogue_messages_comparison.png)

![Average Dialogue Chars](/datasets/toolace_processing_report/toolace_avg_dialogue_chars_comparison.png)

## Что получилось в итоге

На выходе есть четыре итоговых части:

- `multihop_real`: `186`
- `multihop_synthetic`: `5`
- `multihop_total`: `191`
- `singlehop_real`: `204`
- `singlehop_synthetic`: `7,293`
- `singlehop_total`: `7497`