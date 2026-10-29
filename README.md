# Self-evolving dataset

Пайплайн непрерывного обучения одного ученика с меняющимся датасетом: домены,
веса по прогрессу validation loss, выбор окна по C, выход доменов и восстановление
забывания. Семантические операции выполняются через **`codex exec` с входом
ChatGPT**, модель **`gpt-6.1-sol`, `xhigh`, `fast`**. OpenAI API key не используется.

Реализация и принятые допущения: [итоговый отчёт](docs/IMPLEMENTATION_REPORT.ru.md).
Для реального запуска нужны свои `goal.md`, `architecture.md`, исходный JSON/JSONL
и численная реализация ученика. В репозитории есть явный offline demo, а не
результат обучения на пользовательском бенчмарке.

## Быстрый запуск без сети и подписки

Python 3.11+, Linux/macOS. Установка требует сети; сама демонстрация работает offline.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[demo]'
evolving-dataset --config examples/arithmetic/config.toml doctor
evolving-dataset --config examples/arithmetic/config.toml run
evolving-dataset --config examples/arithmetic/config.toml baseline
evolving-dataset --config examples/arithmetic/config.toml evaluate \
  --test examples/arithmetic/test.jsonl
evolving-dataset --config examples/arithmetic/config.toml evaluate \
  --test examples/arithmetic/test.jsonl --method baseline
```

Это реальный маленький byte-level RNN с BPTT и SGD, плюс **явно выбранный**
детерминированный арифметический fixture вместо ChatGPT. Он проверяет исполнение
цикла, а не исследовательскую гипотезу или достижение accuracy 0.9. Выход по плато
разрешён. Никакого автоматического переключения production на demo нет.

## Запуск с ChatGPT и своим учеником

```bash
cp config.example.toml config.toml
cp goal.example.md goal.md
cp architecture.example.md architecture.md
mkdir -p data
# Заполнить goal.md и architecture.md; положить data/seed.jsonl.
# В config.toml указать input_fields, answer_field и checkpoint ученика.
pip install -e '.[hf]'
codex login
evolving-dataset doctor --live
evolving-dataset prepare
evolving-dataset run --max-stages 5
evolving-dataset run                    # продолжить с последнего checkpoint
evolving-dataset baseline
evolving-dataset evaluate --test /path/to/test.jsonl
evolving-dataset report
```

Модель и Fast должны быть доступны именно в используемом CLI и аккаунте.
`doctor` проверяет файлы/зависимости/тип входа; **`doctor --live` делает один
небольшой модельный запрос**. При ошибке доступа или игнорировании Fast операция
останавливается. Другую модель и API-оплату код не подставляет. Альтернативный
установленный бинарник задаётся как `teacher.executable`.

Проверенный вариант с отдельной установкой CLI, без изменения глобального:

```bash
npm install --prefix .tools/codex --ignore-scripts --no-audit --no-fund @openai/codex@0.165.0
# Для config.toml в корне: teacher.executable = ".tools/codex/node_modules/.bin/codex"
```

Для небольшого полного теста с настоящим ChatGPT (делает модельные вызовы по подписке):

```bash
evolving-dataset --config examples/codex-smoke/config.toml run --max-stages 3
```

Все пути в TOML считаются относительно самого файла конфигурации. Файлы
`goal.md` и `architecture.md` целиком, с исходными переносами строк, вставляются
в каждый семантический промпт. Содержимое файлов и конфигурация фиксируются в
manifest; изменение входов требует нового каталога output.

## Формат данных

Одна JSON-запись на строку (`.jsonl`) или массив объектов (`.json`). Например:

```json
{"question":"Сколько всего?","table":[[1,2],[3,4]],"answer":10}
```

```toml
[format]
input_fields = ["question", "table"]
answer_field = "answer"
```

Поля и типы исходных записей сохраняются; ChatGPT возвращает примеры с ответами
в той же схеме. Метаданные домена, C и счётчики хранятся отдельно. Все записи
должны соответствовать общей схеме первого примера. Для вывода схемы вложенные
массивы в первом примере должны быть непустыми. Строковые таблицы тоже допустимы.
В конфигурацию ученика не попадают скрытые метаданные домена.

## Цикл

1. ChatGPT предлагает домены (один концепт/правило на домен). Весь seed dataset
   распределяется по доменам пакетами; недостающие домены добавляются до старта.
   Затем ChatGPT один раз упорядочивает полный список по сложности.
2. ChatGPT отдельно создаёт фиксированную validation-часть каждого домена.
   Точное совпадение входов между train и validation запрещено.
3. Первые k доменов дают равные доли N случайно выбранных примеров.
4. Ученик делает фиксированное число шагов. Батчи выбираются случайно с
   возвращением из текущего stage dataset; повторы считаются предъявлениями.
5. Считаются validation loss ученика и семантическая correctness через ChatGPT.
   Веса пропорциональны `loss_before / loss_after`, с защитой от нулевого loss.
6. Проверяется освоенность пула и выполняются переходы доменов. Все новые домены
   вместе получают 50% N; если старых нет — 100% N. Пулы активных и
   восстанавливаемых доменов пополняются до 2N без целевого C или усложнения.
7. C пересчитывается текущим учеником, пулы сортируются по убыванию C. Начало
   окна — `min(floor(accuracy * pool_size), pool_size - quota)`.
8. Забытые домены дают дополнительно 10–20% N, суммарно не более 20% N.
   Число шагов и размер батча остаются прежними. Когда все статусы `passed`,
   процесс завершается. `--max-stages` только прерывает текущий запуск с
   сохранением состояния, не объявляя эксперимент завершённым.

`N >= 10`, чтобы целочисленно представить квоту восстановления. Число доменов
не ограничено программным параметром; размер контекста и ответа CLI остаются
практическими пределами для каталога доменов.

## C

Для каждого токена **только вопроса/таблицы** берётся последний скрытый вектор
ученика, нормируется до единичной длины и считается среднее косинусное сходство
различных пар. Нет разделения intra/inter, самосходств, ответа, system prompt,
goal.md, architecture.md, padding или специальных токенов.

```text
C = (||sum(normalize(h_i))||² - m) / (m * (m - 1))
```

Это вычисляется за O(mh), без матрицы m×m. При одном токене C=0; пустые/нулевые
векторы считаются ошибкой. Исходный метод:
[Angles Don't Lie / GAIN-RL](https://arxiv.org/abs/2506.02281).
Наше определение — явно изменённый вариант и не доказанная мера сложности.

## Ученики и расширение

- `backend = "hf"`: Hugging Face causal LM, full-parameter SFT, AdamW,
  answer-only cross entropy, greedy decoding. Модель загружается с
  `trust_remote_code=False`; молчаливого обрезания длинных примеров нет.
- `backend = "numpy"`: маленький обучаемый byte RNN для offline demo.
- `backend = "plugin"`, `factory = "your_module:create_student"`: функция
  `create_student(config)` возвращает объект протокола
  [Student](src/evolving_dataset/student.py). Python-модуль должен быть доступен
  в окружении. Это путь для другой архитектуры, RL, LoRA или собственного loss.

Адаптер реализует `train_batch`, `losses`, `predict`, `scores`, `save`, `load`.
`train_batch` возвращает реально обработанное число токенов; `losses` — по одному
конечному неотрицательному значению на пример. `scores` не получает эталонный ответ
в качестве входа модели. Checkpoint должен включать optimizer и RNG, если нужны.

## Сохранение и сравнение

`runs/.../` содержит manifest, неизменную подготовку, кэш запросов/ответов Codex,
снимки ученика и scheduler, счётчики предъявлений/токенов и Markdown-отчёт.
Checkpoint публикуется только после успешного завершения всего этапа; повторный
запуск восстанавливает последний зафиксированный этап. Кэш позволяет повторять
семантические операции после сбоя. Кэш не гарантирует побитовую повторяемость
новых запросов удалённой модели. Одновременная запись в один run запрещена lock.

Baseline стартует с тех же исходных весов, равномерно выбирает домены из общего
итогового банка train-примеров, не применяет окна/смену доменов и получает ровно
то же число предъявлений. Подробности и отличие от online baseline — в отчёте.
Test загружается только отдельной командой после завершения обучения.

Файлы авторизации, реальные данные, личные goal/architecture, логи и веса
исключены из Git. В публичном CI запускаются только offline тесты, без ChatGPT.

## Проверки

```bash
pip install -e '.[demo]'
python -m unittest discover -s tests -v
```

Тесты покрывают численную формулу C и градиенты, распределение квот, окно,
плато/забывание/возврат, общий бюджет baseline, сохранение после сбоев, resume,
неизменность validation, дедупликацию и контракт `codex exec`. Дополнительные
тесты HF автоматически включаются при установленных torch и transformers.

Интеграция CLI основана на официальной документации
[non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode) и
[configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
