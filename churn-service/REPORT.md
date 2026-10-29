## Задание 2.1

https://github.com/Kate-vsn/homework01/actions/runs/40922113562 - ссылка на зеленый прогон
https://github.com/Kate-vsn/homework01/pkgs/container/churn-service - ссылка на страницу пакета, тег есть

## Задание 2.2

https://github.com/Kate-vsn/homework01/actions/runs/40918143591 - прогон с намеренной ошибкой в первом тесте (статус 500 вместо 200 на проверке)
https://github.com/Kate-vsn/homework01/actions/runs/40918949159 - здесь ошибка исправлена и тестовая часть пройдена

## Задание 2.3

https://github.com/Kate-vsn/homework01/actions/runs/40936149462 - указан неверный путь к модели. сначала ошибка происходила на этапе деплоя, а после добавления блока с диагностикой по примеру из курса оказалась на этапе тестов, выдавая, что не удалось загрузить артефакты модели.

https://github.com/Kate-vsn/homework01/actions/runs/40927353474 - указан неверный секрет. упало тоже на деплое, и тот же лог с роллаутом. 