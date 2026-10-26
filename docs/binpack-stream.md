# Потоковый Stockfish binpack reader

Реализован отдельной утилитой `tools/binpack` на Rust с dependency
[`sfbinpack` 0.6.5](https://github.com/Disservin/binpack-rust), upstream revision
`df32a6b22e28354c9f928cdedbe058336d3bb3fa`, GPL-3.0. Cargo.lock утилиты
зафиксирован отдельно; игровой движок не получает этот dependency или Python.
Собственного unsafe в адаптере нет; optional BMI2 feature не включён.

## Запуск

```powershell
cargo build --release --manifest-path tools/binpack/Cargo.toml --target-dir target/binpack
cargo build --release --example nnue
node tools/nnue/download_binpack_sample.mjs
./.venv-nnue/Scripts/python.exe tools/nnue/train_nnue.py --binpack-config target/binpack-samples/t91-smoke.json --epoch-rows 4096 --epochs 2 --batch-size 256 --block-rows 2048 --threads 8 --val-rows 256 --val-chunk 128 --warmup-steps 0 --out runs/t91-smoke
```

Выходной run должен быть новым. Python-окружение описано в
[NNUE training](nnue-training.md). Загрузчик получает только первый **полный**
BINP-блок апрельского T91: 1 048 587 байт. Он требует bounded HTTP Range/206
и прекращает загрузку, если сервер пытается вернуть полный файл вместо диапазона.
Источник, pinned revision, диапазон и локальный SHA256 сэмпла записываются
в `target/binpack-samples/source.json`. Контрольная сумма сэмпла:
`cd93fd9977ae28cedc0ac4419c4f56d0640d1fdf8e00674ac2249002256543fd`.
Хеш **полного** исходника для частичного сэмпла не проверяется. Полные T91,
DFRC и T60T70 файлы этим изменением не скачивались.

После скачивания целого binpack и сверки его опубликованного SHA256 можно
указать его в recipe. Относительные пути разрешаются относительно JSON:

```json
{
  "format": "zenith-binpack-stream-v1",
  "files": ["D:/datasets/test91-2026-04-apr-69gb6p.min-v2.v6.binpack"],
  "source": "adamtwiss/t91-binpacks-filtered@71c0e79b934bd5f05e52119607e22a1c8d377bc4",
  "license": "ODbL",
  "score_num": 1,
  "score_den": 1,
  "score_unit": "raw teacher units; identity scale, calibration required",
  "val_mod": 20,
  "max_cp": 3000,
  "quiet": true,
  "validation_seed": 1
}
```

`--epoch-rows N` задаёт число **принятых** тренировочных записей на эпоху.
Если их недостаточно, это ошибка. Проверочная выборка может быть меньше
`--val-rows`, если пул маленький, но пустой быть не может. Источник, лицензия,
описание единиц и положительная целочисленная шкала обязательны.

## Память и разделение данных

Rust индексирует только BINP-заголовки с проверкой magic, длины и границ файла.
Индекс ограничен миллионом блоков, payload — 100 MiB на блок (обычно около
1 MiB). Блоки перемешиваются по seed эпохи. Внутри блока продолжения
декодируются последовательно; Vec со всеми его распакованными позициями
не создаётся.

Записи передаются бинарными порциями через pipe. Python перемешивает
ограниченный буфер `--block-rows` и выдаёт batches PyTorch. Память зависит
от ограниченного индекса, одного сжатого блока, буферов и модели, а не общего
числа позиций. Полного 100-байтного тренировочного датасета на диске нет.
Это перемешивание блоков и ограниченных буферов, а не равномерная перестановка
всех позиций корпуса; качество/веса смеси нескольких корпусов проверяются отдельно.

Общий encoder с TSV-подготовщиком: `examples/nnue_data/mod.rs`. Split хеширует
96 байт относительных входных признаков. Одинаковый вход, включая цветовые
зеркала или совпадения в других файлах, всегда получает один split. Это
защита от точных совпадений входов, не разделение по партиям/семействам.

## Метки и фильтры

Score в формате относительно **стороны хода**: повторного переворота знака нет.
Raw `abs(score)>=30000` консервативно отбрасывается **до** изменения шкалы,
включая `VALUE_NONE=32002` и матовый/табличный диапазон. Затем применяется
`score_num/score_den` с делением к нулю и проверяется `max_cp`.
Идентичная шкала сохраняет числа учителя, но не доказывает их эквивалентности
нормализованным UCI centipawns. Калибровка конкретного источника остаётся
отдельной проверкой перед серьёзным обучением.

При `quiet=true` пропускаются teacher capture/promotion/en-passant. Обычная
рокировка сохраняется. Это фильтр по **ходу учителя**, а не SEE всех доступных
взятий из TSV-импортёра. Zenith проверяет стандартный FEN, отсутствие шаха,
terminal/явной ничьей, и легальность хода учителя. Result проверяется и
показывается в inspect, но текущий loss обучается только по score.
Поисковой глубины в binpack нет: record.depth=0 означает неизвестную глубину,
а не game ply. Истинные Chess960-рокировки не поддерживаются; совместимость
полного DFRC-корпуса пока не проверена. Строгие правила доски не ослабляются.

## Ошибки и resume

Протокол `ZBSTRM1\0`: ограниченные порции с u32 длиной и 100-байтными записями,
нулевая порция, JSON-статистика и EOF. Python требует правильные размеры,
счётчик, success footer и exit code 0. Короткий заголовок/payload — ошибка.
Если upstream паникует при повреждённых сжатых данных, дочерний процесс
завершается без footer: эпоха не признаётся завершённой и checkpoint не
публикуется. Проверка структуры не заменяет исходные контрольные суммы.

При отмене prefetch убивает декодер и дожидается его завершения. Ограниченная
очередь создаёт backpressure. Diagnostics не смешиваются с бинарным stdout.
Не пропускайте режим `stream` через текстовые PowerShell-команды.

Seed ридера получается из сохраняемого RNG тренера, validation использует
свой фиксированный seed. Resume contract включает recipe, метаданные файлов,
SHA256 ридера/адаптера, настройки и epoch-rows. Многогигабайтные файлы целиком
при каждом запуске не хешируются. При изменении данных/ридера используйте новый
run и `--init-model`. Provenance содержит decoded/emitted и причины пропусков.

Для просмотра исходных данных:

```powershell
./target/binpack/release/zenith-binpack.exe inspect --input target/binpack-samples/t91-april-first.binpack --limit 5 --no-shuffle
```

`pack-fixture OUTPUT.binpack` — вспомогательный упаковщик максимум 8192 строк
`FEN<TAB>UCI<TAB>score_stm<TAB>result_stm` из stdin. Создаёт только новый файл;
используется тестами, а не для bulk-конвертации.

## Проверки

```powershell
cargo test --release --manifest-path tools/binpack/Cargo.toml --target-dir target/binpack
cargo clippy --manifest-path tools/binpack/Cargo.toml --target-dir target/binpack --all-targets -- -D warnings
cargo test --release --example prepare_nnue --test nnue
./.venv-nnue/Scripts/python.exe -m unittest discover -s tools/nnue -p "test_*.py" -v
```

Покрыты compressed chains, обе стороны хода, знаки score/result, рокировка,
EP/превращения, upstream fixed fixture, canonical bytes и split, VALUE_NONE
до rescale, обрезанные файлы, отмена, обучение/экспорт и точное resume.
Оба режима тренера, record и binpack, проходят проверки.

На реальном T91-блоке 50000 принятых записей потребовали 80499 decoded:
23428 sentinel/mate, 4852 score, 1749 other split, 470 teacher tactical.
Один локальный замер reader+IPC+batch assembly на кэшированном блоке занял
0.375 с (~133 тыс. принятых записей/с), без обучения. Это не прогноз скорости
полного корпуса и не оценка Elo.

Проверено 2026-10-26: 13 Python и 13 Rust тестов прошли; fmt и Clippy обоих
Cargo packages чистые. На реальном T91-сэмпле выполнены 2 эпохи по 4096
тренировочных позиций с 256 validation-позициями и 8 потоками PyTorch CPU.
Проверка экспорта на 128 записях × 2 цвета дала 256 полных совпадений между
trainer, NumPy spec, Rust scalar и AVX2 (включая delta/refresh/dense внутри Rust).
Артефакты находятся в `target/binpack-stream-smoke`: это маленький тестовый
run с некалиброванными единицами учителя, а не утверждённая игровая сеть.
