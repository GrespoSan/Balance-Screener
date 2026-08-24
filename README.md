# G. Balance Stock Active V2.1 — Daily chiusa

Screener azionario Daily basato esclusivamente sulle Balance del motore originale.

## Regola base
- Lookback Balance: 500 barre Daily.
- Tocco reale: `low <= top Balance` e `high >= bottom Balance`.
- AREA ATTIVA: almeno 2 delle ultime 3 Daily **chiuse** hanno toccato la stessa Balance.
- Nessuna soglia ATR di vicinanza, nessun WATCH/TRIGGER, nessun LONG/SHORT, nessuno score aggiuntivo.

## Opzione di screening
`Richiedi ultimo Close Daily dentro la Balance`
- ON (default): oltre ai 2 tocchi su 3, il Close dell'ultima Daily chiusa deve essere dentro la fascia.
- OFF: vengono mostrate tutte le Balance con almeno 2 tocchi reali nelle ultime 3 Daily chiuse, anche se l'ultimo Close è già fuori dalla fascia.

La tabella mostra sempre la colonna `Ultimo Close dentro`, così il confronto tra ON/OFF è immediato.

## Daily aperta
La Daily in corso viene esclusa durante la seduta regolare. Dopo la chiusura della seduta, se Yahoo ha già pubblicato la barra finale, essa può diventare l'ultima Daily chiusa usata dalla scansione.

## File ticker
Italia: si possono usare ticker semplici come `ENI`, `UCG`, `ISP`; l'app aggiunge `.MI`.
USA: usare i ticker Yahoo, es. `AAPL`, `MSFT`, `NVDA`.
