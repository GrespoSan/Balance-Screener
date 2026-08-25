# G. Balance Stock Screener V4.1

Modifica tecnica rispetto alla V4.0: controllo freschezza dei dati Daily Yahoo.

- Il download principale resta batch.
- Dopo il batch, ogni ticker viene confrontato con la Daily chiusa più recente disponibile nello stesso mercato.
- Solo i ticker arretrati vengono riscaricati singolarmente.
- Il dataset viene sostituito soltanto se il retry individuale contiene una Daily più recente.
- La tabella mostra `Ultima Daily` vicino al ticker.
- Nel grafico viene mostrata anche la data dell'ultima Daily utilizzata.

Nessuna modifica a motore Balance, dominanza storica S/R, AREA ATTIVA, tocchi, Score o filtro visivo Supporto/Resistenza.
