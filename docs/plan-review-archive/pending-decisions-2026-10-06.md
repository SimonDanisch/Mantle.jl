# Offene Entscheidungen zum Mantle-Umbau, Stand 2026-10-06

Dieses Dokument legt dir 13 offene Design-Fragen vor, jede mit dem Problem und seinen Belegen, den Optionen mit ihren Folgen und einer Empfehlung. Die Fragen 1 bis 12 sind die Liste vom 2026-10-05 (`pending-decisions-2026-10-05.md`). Frage 13 (Mantle lädt nicht ohne Display) ist neu. Bis zu deiner Antwort folgt die neue Suite unter `dev/Mantle/test/suite/areas/` der Empfehlung vom 2026-10-05, und jeder davon abhängige Fall trägt den Kommentar `# pending decision N`. Wo ich die Empfehlung hier ändere, steht dabei, welche Fälle sich ändern. Bitte antworte mit einer Liste wie "1A, 2A, 3A, 4.1A 4.2A 4.3A 4.4A 4.5A, 5.1A …". Die Fragen 4, 5 und 6 bündeln mehrere Unterfragen und brauchen eine Option je Unterpunkt.

Kürzel für Quellen: `api` = `dev/Mantle/docs/api.md`, `plan` = `recording-plan.md`, `int` = `internals.jl`, `rr` = `resizing-and-raytracing.jl` (alle in `dev/Mantle/docs/`, Stand 2026-10-05 18:33), `dec` = `experiments/plan-review/decisions.md`, `cat-core` und `cat-rt` = `catalogue-core.md` und `catalogue-resize-rt.md` im selben Ordner. Suite-Dateien stehen relativ zu `dev/Mantle/test/suite/areas/`.

---

## 1. `persistent = true` für Graph-Arrays

**Problem.** `api:71` bietet `MantleArray(g, …; persistent = true)` an: "Kept between runs, used only by `g`; its own region." Seit B3.16 (`dec:244-254`) sagt dieselbe Datei in `api:111-113` etwas anderes: "A graph array's contents live only during a run; what must survive runs and recompiles (a film, an accumulation count) is a device array the graph names." Im Plan steht derselbe Widerspruch: `plan:195` und `plan:201-203` beschreiben Inhalte, die "persistent on the graph" sind, `plan:440-443` dagegen ein Device-Array. Weitere Stellen auf dem alten Stand:
- `examples.jl:79-80` (`prev = MantleArray(g, Float32, 512; persistent = true)`),
- `int:721` (persistente Graph-Arrays in der Liste geteilter Ressourcen),
- `plan:1897-1901` (`test_runs_never_overlap.jl` mit "a persistent array"),
- `dec:43-44` (B.3, die alte Liste mit `persistent = true`).

Offen ist also, ob ein persistentes Graph-Array eine Neukompilierung überlebt. Für dich hängt daran, ob ein Zähler, Hikaris Film, der rekurrente Zustand eines DNN oder der Vorframe im Editor bei jedem `insert!` oder `delete!` eines Plots erhalten bleibt oder auf Null fällt. Im Katalog betrifft das MUT-14 und Mehrdeutigkeit 4 (`cat-core:74`, `:435`). In der Suite hängen `mutation.jl:300-313` (MUT-14 erwartet einen `MethodError` für das Keyword) und `run.jl:120-147` (RUN-01 zählt in einem Device-Array) daran.

**Optionen.**
- **A. `persistent` streichen.** Was Runs überdauert, ist ein Device-Array, das der Graph benennt:
  ```julia
  prev = MantleArray(dev, Float32, 512)    # statt MantleArray(g, …; persistent = true)
  fill!(prev, 0f0)                          # läuft sofort
  prev .= 0.9f0 .* prev .+ 0.1f0 .* y       # y liegt auf g: ein Knoten von g, prev bekommt einen Hold
  ```
  - Mantle: nur Textänderungen an den sechs Stellen oben, kein neuer Code. Es gibt eine Ressourcenart weniger.
  - Laufzeit: Ein Device-Array hat Holds, `namers` und einen `SyncState`. Ein Run, der es schreibt, liest den `namers`-Snapshot ohne Lock (`plan:599-600`). Vermutung: ein paar ns je geschriebenem Array, nicht gemessen.
  - Gewinn: `Array(prev)` funktioniert. Für ein Graph-Array wirft es (`api:85`).
  - Tests: MUT-14 und RUN-01 bleiben, wie sie sind.
- **B. `persistent` als Region behalten, die dem Graph gehört** (nicht dem Plan) und jede Neukompilierung überlebt.
  - Mantle: eine Liste solcher Regionen am `Graph`, die `recompile!` in den neuen Plan übernimmt und `free!(g)` freigibt.
  - Folge: Das Ergebnis ist ein Device-Array mit Einschränkungen (vom Host nicht lesbar, nicht teilbar). Damit gibt es zwei Wege für denselben Zweck.
  - Tests: MUT-14 prüft dann das Überleben statt des `MethodError`.
- **C. `persistent` behalten, der Inhalt stirbt mit dem Plan.** Es braucht keinen Code zum Übernehmen. Jede Plot-Änderung setzt aber den Zustand zurück, und genau das hat B3.16 für den Film ausgeschlossen ("the film is a device array the screen holds, so accumulation survives plot changes", `dec:252-254`).

**Empfehlung: A** (wie am 2026-10-05). B3.16 nennt Device-Arrays als das, was eine Strukturänderung übersteht. A ist die einzige Option ohne zweiten Weg für denselben Zweck (Plan-Regel 2, "No duplication", `plan:42-43`). Du gibst damit die Kurzschreibweise auf dem Graph auf, und kein Mantle-Mechanismus verhindert mehr, dass ein zweiter Graph das Array benennt.

---

## 2. `when!`-Flags mit anderen Werten als 0 und 1

**Problem.** Für ein Flag mit 0 oder 1 beschreiben `api:130` (Vertrag) und `plan:2289-2291` bzw. `int:396-398` (Mechanismus) dasselbe Verhalten. Für alle anderen Werte widersprechen sie sich:
- Vertrag (`api:130`): "Every launch declared in the block runs only where `flag` is nonzero on the GPU."
- Mechanismus (gleicher Eintrag): "whoever writes their indirect commands multiplies the count by the flag".

Was bei anderen Werten passiert:
- Bei 2 startet der Befehl doppelt so viele Gruppen. Ob die zusätzlichen Gruppen über den `ndrange`-Slot maskiert werden, ist nicht festgelegt.
- Bei -1 als `Int32` ergibt die Umwandlung in den vorzeichenlosen Zählwert eine riesige Gruppenzahl. Vermutung: Timeout oder Device-Verlust.
- Bei `0.5f0` oder `NaN` ist die Multiplikation nicht definiert.

Der Elementtyp des Flags ist nirgends festgelegt ("a `GPURef` or a one-element array", `api:130`). Verwendet werden heute nur 0 und 1:
- Die Gates des Editors sind `UInt32`-Buffer mit 0 oder 1 (`dev/VideoEditor/src/gpugraph.jl:525-527`). Ihr Idiom `repeat!(g, 1; while_nonzero = gate)` soll zu `when!` werden (`plan:373-374`).
- RayMakies `visible` ist ein `MantleArray{UInt32,1}` (`rr:1435`, `:1447`).

Vom Kernel geschriebene Flags (Readback) sind der Fall, in dem ein Wert über 1 versehentlich entsteht. Im Katalog betrifft das WHEN-04 und Mehrdeutigkeit 6 (`cat-core:119`, `:437`), in der Suite `when.jl:150-165` (WHEN-04: Flags 2, -1, `typemax(Int32)`, `0.5f0`, `NaN32`, je zwei Runs, zwei Inkremente).

**Optionen.**
- **A. Ungleich 0 heißt "läuft einmal".** Head- und Prepare-Kernel schreiben `count * (flag != 0)`.
  - Kosten: ein Vergleich in Kerneln, die ohnehin laufen. Nichts kompiliert neu.
  - Gleiche Regel wie `while_nonzero` (`api:129`). CUDA und HIP prüfen das Flag nativ (IF-Knoten bzw. eine Prüfung am Kernel-Eingang, `plan:1492`, `int:403-405`). Vermutung, nicht nachgeprüft: CUDAs IF-Knoten führt seinen Körper bei einem Wert ungleich 0 aus.
  - Bei Floats läuft `NaN` (weil `NaN != 0`), `-0.0` läuft nicht.
  - WHEN-04 bleibt.
- **B. Wie A, aber nur ganzzahlige Flags** (`Bool`, `UInt32`, `Int32`). Ein Float-Flag wirft bei `when!`. Damit entfällt die Frage nach `NaN` und `-0.0`, dafür kostet es eine Prüfung bei der Deklaration. In WHEN-04 erwarten die Fälle `0.5f0` und `NaN32` dann einen `ArgumentError`.
- **C. "count × flag" wörtlich nehmen**, andere Werte als 0 und 1 sind undefiniert (dokumentiert). Das kostet keinen Code. Ein Kernel, der versehentlich eine Länge statt 0/1 in das Flag schreibt, startet still falsch viele Gruppen. WHEN-04 entfällt.

**Empfehlung: A** (wie am 2026-10-05). `when!` bekommt dieselbe Regel wie `while_nonzero`, das Editor-Idiom zieht unverändert um, und der Preis ist ein Vergleich ohne Neukompilierung. B verbietet nur Float-Flags; dafür sehe ich keinen Nutzen, der eine eigene Prüfung rechtfertigt. Du gibst die Möglichkeit auf, das Flag als Multiplikator zu verwenden, und die hat niemand verlangt.

---

## 3. Eager-Aufrufe in `when!`- und `repeat!`-Blöcken

**Problem.** Wo ein Aufruf läuft, entscheiden seine Argumente (`api:127`, `plan:342-345`): Sind alle Argumente Device-Arrays, läuft er sofort. Ein `fill!(a, 0)` mit einem Device-Array `a` innerhalb von `when!(g, flag) do … end` oder eines `repeat!`-Körpers läuft deshalb einmal bei der Deklaration, unabhängig vom Flag. `api:130` ("Graphs only: an eager call runs.") und `plan:372` sagen das nur indirekt.

Für dich bedeutet das zum Beispiel:
- In einer Decoder-Schleife (`repeat!`) leert `fill!(kvcache, 0)` auf einem Device-Array den Cache einmal beim Aufbau statt in jeder Iteration.
- In einem Schattenpass von RayMakie ignoriert ein Eager-Aufruf im `when!`-Block das Flag.

Beides passiert still. Für `fill!` gibt es keine Graph-Form; nur `dispatch!(g, kernel, …)` deklariert auch mit reinen Device-Arrays (`int:222`). Im Katalog betrifft das DECL-11 und Mehrdeutigkeit 7 (`cat-core:95`, `:438`), in der Suite `declaration.jl:230-247` (DECL-11 erwartet `ArgumentError` für beide Blockarten, und es läuft nichts).

Die Schwierigkeit liegt bei der Erkennung. Das eager `dispatch!(dev, …)` (`plan:939-956`) kennt keinen Graph. Um zu werfen, muss es fragen, ob die aufrufende Task gerade einen Körper deklariert, also nach `current_task()` suchen. Das tun bereits `lockgraph` (`int:1019-1028`) und `inhostfunction` (`rr:484`, ein Durchlauf über `livegraphs(REGISTRY)`). A.3 (`dec:20-21`) und Plan-Regel 5 (`plan:50-53`) verbieten Zustand, der implizit gefunden wird.

**Optionen.**
- **A. Werfen bei der Deklaration.** `when!` und `repeat!` setzen schon `g.inbody` (`int:358`, `:379`). Der eager Pfad prüft, ob die aktuelle Task in einen Graph mit `inbody` deklariert.
  - Umsetzung: wie `inhostfunction`, oder mit einem Zähler offener Körper je Device, sodass der Normalfall ohne offenen Körper ein atomares Lesen ist. Vermutung: weit unter den 23.5-24 µs eines Submits (`rr:81-83`).
  - Die Prüfung entscheidet nur, ob geworfen wird. Wo ein Aufruf läuft, ändert sie nie.
  - DECL-11 bleibt.
- **B. Stand belassen und dokumentieren**: Der Aufruf läuft einmal bei der Deklaration. Das kostet keinen Code, die stillen Fehler bleiben möglich. DECL-11 erwartet dann "läuft einmal".
- **C. Im Körper deklarieren Eager-Aufrufe in den Graph des Blocks** (die Device-Arrays werden benannt, der Start bekommt das Flag).
  - Für dich am bequemsten. Der Aufruf müsste seinen Graph aber über die aktuelle Task finden, also genau die implizite Suche, die A.3 verbietet, und das Verhalten würde sich dadurch ändern.
  - `fill!(a, 0)` hieße innerhalb und außerhalb eines Blocks verschiedene Dinge.

**Empfehlung: A** (wie am 2026-10-05). Ein lauter Fehler ist besser als ein Aufruf, der still nur einmal läuft. Die Suche nach der Task dient nur einer Prüfung, wie die, die die Doku für Host-Funktionen schon verwendet (`rr:484`). Du gibst damit eine Prüfung mehr im eager Pfad auf, und ein Aufruf, der wirklich "einmal beim Aufbau" gemeint ist, muss vor den Block wandern. Die Fehlermeldung sollte auf `dispatch!(g, …)` verweisen.

---

## 4. Fenster: die Ränder von B3.8

**Problem.** B3.8 (`dec:185-192`) legt fest, dass der Graph, der `render!` in ein Fenster deklariert, es präsentiert, und dass ein zweiter Graph, der hineindeklariert, wirft. Fünf Ränder sind offen (Mehrdeutigkeiten 13-15, `cat-core:444-450`). In der Suite hängen daran:
- `windows.jl:192` (WIN-02), `:203` (WIN-03), `:268` (WIN-07), `:409` (WIN-15), `:425` (WIN-16), `:543` (WIN-24),
- `scheduling.jl:230` (SC-09).

Die Unterpunkte im Einzelnen:
- **4.1 Wer schreibt ins Fenster?** B3.8 sagt "the graph that declares `render!`". `examples.jl:490` lässt aber einen Kernel (`blurpass!(win, blur)`) das Swapchain-Bild schreiben. `plan:1023` ("A graph presents to at most one window") sagt nicht, wo der Fehler entsteht.
- **4.2 Eager in ein Fenster rendern.** `api:175` erlaubt `render!(g or dev, targets...)` mit einem Fenster als Ziel. Ein eager Aufruf hat aber keinen präsentierenden Graph und keinen Run; Acquire, Present und das Überspringen bei minimiertem Fenster passieren in `run!` (`int:1529-1541`).
- **4.3 Ende der Bindung.** Das Fenster hält seinen Graph schwach (`int:1461`). Ob ein anderer Graph nach `free!(g)`, nach dem Einsammeln von `g` oder nach dem Löschen des Render-Knotens hineindeklarieren darf, ist offen (`cat-core:448`).
- **4.4 `Array(win)` ohne folgenden Run.** `api:184` wartet auf "the next run presenting to the window". Wenn das Fenster minimiert ist (Runs werden übersprungen, `api:483`), es keinen präsentierenden Graph mehr gibt, oder der Aufruf aus einer Host-Funktion des präsentierenden Graphen kommt, wartet er ewig.
- **4.5 Rückgabewert von `run!`.** `api:101` nennt keinen. `int:1031-1054` gibt `Submitted()` oder `Skipped()` zurück und sagt "both public" (`int:1054`), `api` führt beide Typen aber nicht auf. Ohne sie kann der Aufrufer einen übersprungenen Run (Ausgaben veraltet, Änderungen noch ausstehend) nicht von einem echten unterscheiden (`cat-core:450`).

**Optionen.**
- **4.1 Wer schreibt ins Fenster**
  - **A.** Jeder Knoten des präsentierenden Graphen darf es schreiben (`render!`, ein Kernel, eine Kopie). Der erste Graph, der irgendeinen Schreibzugriff deklariert, wird zum präsentierenden Graph. Folgende Deklarationen werfen:
    - ein Schreibzugriff aus einem anderen Graph,
    - ein zweites Fenster im selben Graph.

    WIN-02 und WIN-03 bleiben, wie sie sind.
  - **B.** Nur `render!` darf ins Fenster schreiben. Ein Kernel, der das Fenster schreibt, wirft. Nachbearbeitung schreibt dann in ein `Image`, danach folgt ein Pass oder eine Kopie ins Fenster. Das kostet eine bildschirmgroße Kopie pro Frame, laut `plan:997-1001` geschätzt 0.08 ms auf der RX 7900 XTX und 0.21 ms auf der RTX 4000 Ada bei 3840×2160. WIN-03 ändert sich.
- **4.2 Eager `render!(dev, win => …)`**
  - **A.** Wirft beim Aufruf. Eager in `Image`s zu rendern bleibt erlaubt.
  - **B.** Erlaubt, solange kein Graph das Fenster präsentiert: Acquire, Rendern und Present in einer Submission. Das braucht neben B3.8 ein zweites Besitzmodell für Fenster, Acquire außerhalb von Graphen und ein eigenes Verhalten bei minimiertem Fenster für eager Aufrufe.

  Es gibt dazu noch keinen Suite-Fall; PIECE-20 (`pieces.jl:533-545`) prüft nur `insert!` in einen eager Pass. Ein Fall ist anzulegen.
- **4.3 Ende der Bindung**
  - **A.** Die Bindung endet in drei Fällen. Danach darf ein anderer Graph hineindeklarieren (WIN-24).
    - mit `free!(g)`,
    - wenn `g` eingesammelt wird,
    - wenn `g` keinen Knoten mehr hat, der das Fenster schreibt (`delete!(g, h)` des Blocks). Das wirkt erst, sobald der alte Plan ausgemustert ist, weil der noch präsentieren kann.
  - **B.** Die Bindung gilt, solange das Fenster lebt. Das ist einfacher, weil es keinen Freigabepfad gibt. Baut RayMakie aber den Frame-Graph eines Screens für dasselbe GLFW-Fenster neu, muss es das Fenster schließen und neu öffnen.
- **4.4 `Array(win)` ohne folgenden Run**
  - **A.** Wirft `ArgumentError` in den drei Fällen (minimiert, kein präsentierender Graph, aus der eigenen Host-Funktion). Ein Aufruf, der schon wartet, wird mit dem Fehler geweckt, wenn der nächste Run des Graphen übersprungen wird oder der Graph freigegeben wird. WIN-15 und WIN-16 bleiben.
  - **B.** Warten. Bei minimiertem Fenster kann das ewig dauern, aus der eigenen Host-Funktion ist es ein Deadlock.
  - **C.** Das zuletzt präsentierte Bild zurückgeben. Dafür muss jeder präsentierte Frame in Readback-Speicher kopiert werden, mit denselben Kosten wie 4.1 B pro Frame.
- **4.5 Rückgabewert von `run!`**
  - **A.** `Submitted()` und `Skipped()` werden öffentlich und in `api` 2.3 aufgeführt. RayMakie kann dann zum Beispiel `colorbuffer`-Reads überspringen. WIN-07 und SC-09 bleiben.
  - **B.** `run!` gibt `nothing` zurück, dazu kommt eine Abfrage am Fenster (Name zu wählen). Abfrage und Run können sich widersprechen, wenn sich das Fenster dazwischen ändert.
  - **C.** `true`/`false`. Das ist dieselbe Information wie A, lässt aber keinen Platz für weitere Zustände.

**Empfehlung: 4.1 A, 4.2 A, 4.3 A, 4.4 A, 4.5 A** (wie am 2026-10-05; neu ist in 4.4 das Wecken wartender Aufrufe):
- **4.1 A**, weil die Fehler, die B3.8 beseitigt hat, von mehreren präsentierenden Graphen kamen, nicht von Kerneln. B würde also pro Frame kosten, ohne einen dieser Fehler zu verhindern.
- **4.2 A**, weil B das Besitzmodell wieder aufmachen würde, das B3.8 geschlossen hat.
- **4.3 A**, weil RayMakie sonst Fenster neu öffnen müsste.
- **4.4 A**, weil die anderen Optionen hängen bzw. ein Vollbild pro Frame kopieren.
- **4.5 A** steht schon so in `int`. Es fehlt nur der Eintrag in `api`.

Du gibst damit auf: eager Rendern direkt ins Fenster (etwa für einen Startbildschirm ohne Graph).

---

## 5. Read-only-Eingaben (B3.9)

**Problem.** B3.9 (`dec:193-198`) verlangt für die Eviction zwei Eingabearten:
- Ein Array aus einer Datei darf auch den RAM verlassen; beim Restore wird die Datei neu gelesen.
- Ein Array aus Host-Daten verlässt nur den VRAM.

Name ("`immutable`?") und Durchsetzung sind offen; der Inhalt "must not change". `plan:2335-2338` führt die Eingabearten als noch zu klärend. `api:69` schreibt `MantleArray(path; mmap = true)` ohne Device und verletzt damit Plan-Regel 5 (`plan:50-53`).

Die Suite setzt die Empfehlung vom 2026-10-05 in zwei verschiedenen Formen um:
- `creation.jl:57` und `eviction.jl:100`: `MantleArray(d, path; mmap = true, eltype = T, dims = dims)`,
- `dnn_patterns.jl:323`: `MantleArray(dev, path, Vector{Float32}, (n,); mmap = true)`.

Wen es betrifft: DNN-Gewichte. B3.6 verlangt, dass zwei Modelle mit je 90 % VRAM beide laufen (`dec:168-169`; EVICT-01 in `eviction.jl:122`, DNN-06 in `dnn_patterns.jl:350`). Ein normales Array wird bei jeder Eviction hinaus- und beim Restore wieder hineinkopiert. Ein read-only-Array braucht nur die Kopie beim Restore. Vermutung, abgeleitet, nicht gemessen: halber PCIe-Verkehr, wenn zwei Modelle sich abwechseln.

Katalog: Mehrdeutigkeit 16 (`cat-core:451-455`), EVICT-23 bis 26 (`cat-core:343-346`), DATA-08, DATA-10, CR-04. Suite:
- `creation.jl:167` (DATA-08), `:183` (DATA-10), `:232` (CR-04),
- `dnn_patterns.jl:311-336` (DNN-05),
- `eviction.jl:476` (EVICT-23), `:495` (EVICT-24), `:511` (EVICT-25), `:533` (EVICT-26).

Die fünf Unterpunkte:
- **5.1 Name.**
- **5.2 Durchsetzung.** Mantles eigene Schreibmethoden kennen ihr Ziel beim Aufruf. Ein roher Kernel `dispatch!(g, k, (a, …), n)` schreibt dagegen, was sein Code schreibt. Das zu wissen braucht den Zugriff je Argument (G2, `api:549-556`, ein Vorschlag), und Graph-Kernel kompilieren erst beim Compile, nicht bei der Deklaration (`api:310`). EVICT-25 (`eviction.jl:511-530`) erwartet den Fehler aber schon bei `dispatch!(g, …)`.
- **5.3 Host-Daten: woher stellt der Restore wieder her?** "Leaves only VRAM" heißt, dass Mantle den Inhalt im RAM hält. Nach B3.18 (`dec:269-277`) streamt die Erzeugung in Chunks und behält nichts, und der Aufrufer darf seine Daten danach ändern (CR-05). EVICT-24 (`eviction.jl:495-509`) erwartet eine Eviction ohne jede Kopier-Submission.
- **5.4 Signatur der Dateiform.**
- **5.5 Die Datei ändert sich auf der Platte.**

**Optionen.**
- **5.1 Name**
  - **A.** Das Keyword `readonly = true` an `MantleArray(dev, data; …)`. Die Dateiform ist immer read-only. Es bedeutet dasselbe wie SPIR-V `NonWritable` und KAs `@Const`.
  - **B.** `immutable = true` (B3.9s Vorschlag). In Julia ist Unveränderlichkeit eine Eigenschaft eines Typs (`ismutable`), nicht des Inhalts. Das lädt zur Verwechslung mit `struct` ein.
- **5.2 Durchsetzung**
  - **A.** Mantles eigene Schreibmethoden werfen beim Aufruf:
    - Stores,
    - `copyto!` und `copy!`,
    - `resize!` und `fill!`,
    - Broadcast-Ziele,
    - Graph-Kopien.

    Rohe Kernel prüft Mantle über G2 beim Compile. Es wirft also `run!(g)`, bzw. das eager `dispatch!(dev, …)`, das beim Aufruf kompiliert. In EVICT-25 wird die Zeile `dispatch!(g, …)` zu `@test_throws ArgumentError run!(g)`.
  - **B.** Wie A, aber Graph-Kernel kompilieren schon bei der Deklaration, damit `dispatch!(g, …)` wirft. Die Kompilierzeit wandert von `run!` zur Deklaration, insgesamt gleich viel (Vermutung). Kernel von Graphen, die nie laufen, werden trotzdem kompiliert.
  - **C.** Ein read-only-Typ auf Device-Seite (`devicearg` ohne `setindex!`): Ein Schreibzugriff scheitert beim Kompilieren des Kernels. Das kostet eine zweite Spezialisierung jedes Kernels, der denselben Typ einmal schreibbar und einmal read-only sieht. Das sind mehr Kompilierungen, gegen "wenig recompile" (B3.15, `dec:225-226`).
  - **D.** Keine Durchsetzung, nur Doku. Ein Schreibzugriff fällt erst auf, wenn der nächste Restore den alten Inhalt zurückbringt.
- **5.3 Host-Daten: woher stellt der Restore wieder her**
  - **A.** Mantle hält ab der Erzeugung eine eigene Host-Kopie. Eviction kopiert nie hinaus.
    - RAM: die volle Array-Größe, solange das Array lebt, auch wenn es nie verdrängt wird.
    - Die Erzeugung kopiert zweimal.
    - EVICT-24 bleibt.
  - **B.** Bei der ersten Eviction einmal hinauskopieren und die Host-Kopie danach behalten.
    - RAM erst, wenn tatsächlich Druck auftrat.
    - Höchstens eine GPU-zu-Host-Kopie pro Array.
    - EVICT-24 ändert sich: Die erste Eviction macht eine Submission, die zweite keine.
  - **C.** Keine Host-Daten-Art. Nur die Dateiform ist besonders, read-only-Arrays aus Host-Daten werden wie jedes Array verdrängt (jedes Mal hinauskopiert). EVICT-24 entfällt.
- **5.4 Signatur der Dateiform**
  - **A.** `MantleArray(dev, path::AbstractString, T, dims...; offset = 0)`. Gleiche Form wie `MantleArray(dev, T, dims...)`, Dispatch auf den Pfad, kein `mmap`-Keyword, immer read-only.
  - **B.** `MantleArray(dev, path; mmap = true, eltype = T, dims)` (wie `creation.jl:57`, `eviction.jl:100`).
  - **C.** `MantleArray(dev, path, Vector{T}, dims; mmap = true)`, angelehnt an `Mmap.mmap(io, Array{T,N}, dims)` (wie `dnn_patterns.jl:323`).

  Jede Option korrigiert `api:69` (Device als Argument). Eine der beiden Suite-Formen ändert sich in jedem Fall.
- **5.5 Die Datei ändert sich auf der Platte**
  - **A.** Undefiniert und dokumentiert. Der Restore läuft durch und das Device bleibt benutzbar (EVICT-26, `eviction.jl:533-550`).
  - **B.** Mantle merkt sich Größe und mtime bei der Erzeugung und vergleicht bei jedem Restore (ein `stat`). Eine Änderung lässt den Run werfen, der wiederherstellt.
    - Das erkennt eine ersetzte Datei, aber kein Überschreiben am Platz innerhalb der mtime-Auflösung.
    - Es ist ein Vergleich, wie ihn A.4 (`dec:22-24`) für Mantles eigene Änderungen ausschließt. Eine Mantle-Funktion, die die Änderung melden könnte, gibt es hier allerdings nicht.
  - **C.** Die Datei bei der Erzeugung in den RAM kopieren. Mantle hängt dann nicht mehr von der Datei ab, aber das "can leave even RAM" aus B3.9 fällt weg.

**Empfehlung: 5.1 A, 5.2 A, 5.3 B, 5.4 A, 5.5 A.**
- **5.2 A:** Mantles eigene Schreibmethoden kennen ihr Ziel beim Aufruf und werfen dort. Für rohe Kernel geht das bei Graph-Kerneln frühestens beim Compile. Die Empfehlung vom 2026-10-05 ("wirft bei der Deklaration") gilt also nur für Mantles eigene Methoden; EVICT-25 ändert sich in einer Zeile.
- **5.3 B weicht von der Suite ab**, die A angenommen hat. A verdoppelt den RAM aller read-only-Gewichte dauerhaft, gegen "no waste" (B3.6, `dec:163-164`). B kostet höchstens eine Kopie pro Array.
- **5.4 A** ist eine Namensfrage; ich nehme die Form, die zu `MantleArray(dev, T, dims...)` passt.
- **5.5 A**, weil B nur einen Teil erkennt und eine Prüfung einführt, die A.4 vermeiden will.

Du gibst damit auf: dass rohe Kernel schon bei `dispatch!(g, …)` scheitern, und eine Eviction ohne jede Kopie beim ersten Mal.

---

## 6. Längen, die ein Kernel schreibt

**Problem.** Gemeint ist ein Array mit `capacity = n`, dessen Länge ein Kernel über `lengthof(a)` (Name offen) schreibt (`api:76`, `:90`; `rr:113-116`). `size(a)` und `length(a)` liefern laut `api:86` "the host view". Für `alive = MantleArray(dev, Point3f, 0; capacity = 1_000_000)` (`rr:115`) bleibt diese Sicht 0, egal was der Kernel schreibt. Die Länge auf dem Device liefert `devicesize(a)` (Name offen, `api:87`). `lengthsource` und `capacity` sind Konstanten (`rr:216-217`), und Temporaries über so ein Array werden aus der Kapazität dimensioniert (`rr:800-802`).

Offen ist (`cat-rt:432-438`, A13):
- was `length` liefert,
- was `resize!` und `copy!` tun,
- was passiert, wenn ein Kernel mehr als die Kapazität schreibt.

Es betrifft Hikaris Work-Queues (`rr:1413-1414`), Partikelsysteme und die Simulation, die ohne Kopien gezeichnet wird (`plan:1650-1657`). Suite: `lengthof.jl:4-8` (Kopf), `:91` (KL-03), `:121` (KL-05), `:169` (KL-08).

Die drei Unterpunkte:
- **6.1** Was liefern `length` und `size`?
- **6.2** Was tun `resize!` und `copy!`? `resizeto!` (`rr:540-551`) setzt bei jedem Array die Dims und vermerkt ein Resize. Bei einem solchen Array würde ein Resize über die Kapazität den Speicher verschieben, während `capacity` (konstant) und die daraus dimensionierten Temporaries gleich blieben. Spätere Starts würden dann über das Ende hinaus schreiben.
- **6.3** Was passiert, wenn ein Kernel mehr als die Kapazität schreibt?

**Optionen.**
- **6.1 `length`/`size`**
  - **A.** Werfen. Stattdessen gibt es `devicesize(a)` und `capacity(a)`. Folgen:
    - Basis-Broadcasting fragt `axes(dest)` ab und prüft die Achsen der Argumente (Julia 1.12.7, `base/broadcast.jl:905`, `:316`). `y .= alive .* 2f0` braucht deshalb eine Mantle-Methode, die Base für solche Arrays nicht nach Achsen fragt; Mantle startet ohnehin über `launchrange`.
    - `show` darf `length` nicht aufrufen.
    - KL-03 bleibt.
  - **B.** Die Kapazität liefern. Generischer Code läuft, und auf dem Host dimensionierte Temporaries sind groß genug (wie `SizedBy`, `rr:802`). Aber `for i in 1:length(a)` und `similar(a)` arbeiten still auf veralteten Elementen hinter der Device-Länge, und `length(Array(a)) != length(a)`.
  - **C.** `devicesize(a)` liefern, also bei jedem Aufruf eine Submission und ein Warten. Das ist korrekt, aber jeder `length`-Aufruf in generischem Code kostet einen GPU-Roundtrip, und Deklarationen würden Device-Zustand lesen.
- **6.2 `resize!`/`copy!`**
  - **A** (2026-10-05). Beide werfen. Dann gibt es keinen Weg vom Host, die Länge zu setzen. Ein Array aus Host-Daten anlegen oder einen Partikelzähler auf 0 setzen geht nur über einen Kernel-Start (`dispatch!(dev, setlength!, (lengthof(a), 0), 1)`).
  - **B.** Innerhalb der Kapazität:
    - `resize!` wird ein ausstehender Cell-Write in der Reihenfolge der Aufrufe. Ein späterer `lengthof`-Write eines Kernels im Run überschreibt ihn (so hatte die Katalogzeile von KL-08 es zuerst angenommen, `cat-rt:168`).
    - `copy!(a, data)` ist dieser Cell-Write plus ein Store.
    - Stores werden gegen die Kapazität geprüft.

    Über die Kapazität hinaus werfen beide.
  - **C.** Auch über die Kapazität hinaus erlaubt. Die Kapazität wächst, und Pläne mit daraus dimensionierten Temporaries werden stale und kompilieren einmal neu. Das ist die Regel für Temporaries, die für weniger Elemente dimensioniert sind (`rr:664-667`).
- **6.3 Ein Kernel schreibt mehr als die Kapazität**
  - **A.** Der Prepare-Knoten kappt den Wert auf die Kapazität, sowohl für die Start-Befehle als auch in der Cell (KL-05: `devicesize == cap`). Das kostet ein `min` in einem Kernel, der ohnehin läuft. Was der fehlerhafte Kernel über die Kapazität hinaus geschrieben hat, liegt schon außerhalb des Speichers; das Kappen schützt nur die folgenden Starts.
  - **B.** Kappen und zusätzlich eine Overflow-Marke, die der Host lesen kann (Name zu wählen). Das braucht einen Readback-Slot pro Array und neue API.
  - **C.** Nichts festlegen. Spätere Starts über die Cell lesen und schreiben dann hinter den Speicher. Vermutung: Device-Verlust auf RADV, wie bei anderen Zugriffen außerhalb der Grenzen.

**Empfehlung: 6.1 A, 6.2 B, 6.3 A.**
- **6.1 A** und **6.3 A** wie am 2026-10-05: ein lauter Fehler statt stiller falscher Längen, und das Kappen kostet ein `min`.
- **6.2 ändere ich von A auf B.** A lässt keinen Weg vom Host, ein solches Array zu befüllen oder zurückzusetzen. B braucht keine neue Schrittart, `CellWrite` gibt es schon (`rr:559`).

Was sich an KL-08 ändert:
- `resize!(alive, 7)` vermerkt einen Cell-Write, der Kernel des Runs schreibt danach 300, also ist `devicesize == 300`.
- `resize!(alive, cap + 1)` wirft und vermerkt nichts.

Du gibst damit auf: `length` auf solchen Arrays überhaupt, und Mantle muss die Broadcast-Achsenprüfung für sie selbst übernehmen.

---

## 7. Konsistenz innerhalb eines Runs

**Problem.** Die Invariante in `api:452` lautet: "every run sees the old or the new state, never a mix". Ein Run mit Host-Knoten ist aber eine Submission pro Stage, und nach `plan:661-665` wird "a change queued while a host function runs (a `resize!`, a store) ... ordered before the later stages and seen by them". `int:1368-1371` sagt dasselbe. Ausgenommen sind schon jetzt Änderungen, denen der Plan mitten im Run nicht folgen kann (direkte Zählungen, Temporaries); sie bleiben bis zum Ende des Runs ausstehend (`plan:672-686`).

Ein Beispiel: Ein Graph schreibt in beiden Stages über ein Array-`GPURef`, und zwischen den Stages kommt `r[] = b`. Stage 1 schreibt dann `a`, Stage 2 schreibt `b` (RN-18).

Daran hängt auch, was B3.15 mit "waits for that plan's complete last run" (`dec:234-237`) meint, wenn der andere Plan gerade zwischen zwei Stages steht. RN-17 nimmt an:
- Der Schreiber wartet nur auf der GPU auf die schon abgeschickten Stages, ohne Host-Wartezeit.
- Stage 2 des Lesers sieht den neuen Schreibzugriff.

Katalog: A8 (`cat-rt:422`). Suite: `changes_during_runs.jl:164` (RN-01), `:199` (RN-03), `:474` (RN-17), `:491` (RN-18). Ebenfalls abhängig, aber nicht markiert: `host.jl:245-258` (HOST-09, eine Host-Funktion speichert in ein Device-Array, Stage 2 sieht es).

**Optionen.**
- **A. "Alt oder neu, nie gemischt" gilt pro Stage.** Jede Stage ist eine Submission und sieht einen Zustand. Zwischen den Stages dürfen Host-Funktionen und andere Tasks etwas ändern.
  - Mantle: `api:452` und `plan:1904` umformulieren. Die Implementierung in `int` ist schon so.
  - Ein Schreiber, der auf einen Plan zwischen zwei Stages trifft, wartet auf der GPU auf dessen abgeschickte Stages, ohne Host-Wartezeit.
  - RN-01/03/17/18 und HOST-09 bleiben.
- **B. Pro Run.** Jede Änderung, die eintrifft, während ein Plan zwischen Stages steht, und etwas berührt, das er benennt, bleibt bis zu seiner letzten Stage liegen (wie die Stale-Änderungen, `plan:672-686`).
  - Andere, die sie anwenden würden (eager Aufrufe, Runs anderer Graphen), warten auf dem Host den ganzen Run ab (wie `WaitFor`), oder der Plan braucht eine eigene Kopie der Cells.
  - Eine Host-Funktion kann ihren späteren Stages keine Daten mehr per Store übergeben und muss `writes` verwenden. HOST-09 kippt.
  - RN-01/03/17/18 ändern sich; RN-17 bekommt eine Host-Wartezeit.
- **C. Pro Run, streng.** Änderungen an allem, was der laufende Plan benennt, werfen in seinen Host-Funktionen und lassen andere Tasks warten. Wie B, mit mehr Fehlern.

**Empfehlung: A** (wie am 2026-10-05). A ist Plan-Regel 8 ("was nach einer Änderung abgeschickt wird, sieht sie", `plan:59-71`), angewandt auf Stages, die Submissions sind (B3.3, `dec:144-151`). B und C bringen Host-Wartezeiten für andere Graphen während langer Host-Funktionen, und die hat B3.3 nur für geteilte Arena-Teile akzeptiert. Neu ist hier: Mit A legst du auch fest, dass "complete last run" in B3.15 für einen Plan zwischen Stages "seine bis jetzt abgeschickten Stages" heißt.

Du gibst damit auf: dass ein Run mit Host-Knoten eine Momentaufnahme ist. Wer so einen Graph liest, darf nicht annehmen, dass beide Stages dieselbe Eingabe gesehen haben.

---

## 8. Was das Gate "zero compiles" zählt

**Problem.** Die Vorgaben:
- B3.6 (`dec:157-160`): RayMakie "should NOT recompile plans often - only if new plots are added or removed".
- B3.17 (`dec:255-268`): Plots sind Pieces, ein Insert kompiliert keinen Plan, "insert im milisecunden bereich ist ok".
- Das Gate in `api:465` verlangt bei Plot-Inserts und -Deletes "zero compiles".

Ein Piece einer neuen Art braucht aber eine Pipeline, die noch nicht im Device-Cache ist (`compiledraw`, `api:312`; `int:930-931`: "pipelines from the device type's cache"). Dazu kommen drei Lücken:
- Der Kompilierzähler je Device (`api:310`) summiert nur das `compiled::Bool` aus `compilekernel`. `compiledraw` und `compiletrace` (`api:312-313`) liefern kein solches Flag.
- Eine Neukompilierung des Plans, deren Kernel alle aus dem Cache kommen, setzt nirgends `compiled`. Der Zähler sieht sie also gar nicht.
- `api:492` sagt noch, ein Wechsel des Pipeline-Keys "counts as removing and adding the plot: a recompile". Für Raster-Plots ist das unter B3.17 keine Neukompilierung des Plans mehr.

Katalog: Mehrdeutigkeiten 3 und 26 (`cat-core:434`, `:471`). Suite:
- `pieces.jl:155` (PIECE-01: keine Plan-Kompilierung, höchstens eine Pipeline-Kompilierung für das erste Piece, keine für ein zweites derselben Art),
- `:280` (PIECE-08),
- `raymakie_patterns.jl:334` (RM-06: keine Pipeline-Kompilierung, weil die Art schon in Gebrauch ist).

**Optionen.**
- **A. Getrennte Zähler** für Plan-Kompilierungen, Records, Pipeline-Kompilierungen und Kernel-Kompilierungen.
  - Das Gate verlangt null Plan-Kompilierungen bei Insert, Delete und Attributänderungen.
  - Eine Pipeline-Kompilierung ist erlaubt, wenn eine Art zum ersten Mal auf einem Device benutzt wird, nie für einen zweiten Plot derselben Art.
  - Verb-Änderung: `compiledraw` und `compiletrace` liefern `(native, compiled::Bool)` wie `compilekernel` (`api` 4.4).
  - `api:465` und `:492` werden neu geschrieben.
- **B. Streng null, auch für Pipelines.** RayMakie kompiliert jede Pipeline-Variante beim Anlegen eines Screens vor. Die Varianten vervielfachen sich nach Plot-Typ, Argumenttypen (`Point2f`/`Point3f`, Farbtypen), indiziert oder nicht, Texturen und Zielformat (`plan:2378-2382` listet, was heute eine Pipeline auswählt). Vollständiges Vorkompilieren kostet Startzeit für Varianten, die nie benutzt werden; teilweises erfüllt das Gate nicht.
- **C. Ein Zähler für alles**, das Gate erlaubt genau eine Kompilierung pro neuer Art. Damit lässt sich eine Plan-Kompilierung nicht von einer Pipeline-Kompilierung unterscheiden, und eine Plan-Kompilierung mit Kerneln aus dem Cache bleibt unsichtbar (siehe oben).

**Empfehlung: A** (wie am 2026-10-05), zusätzlich mit der Verb-Änderung an `compiledraw`/`compiletrace`, ohne die das Gate Pipelines gar nicht zählen kann. Nur A unterscheidet den teuren Fall (Plan) vom einmaligen Fall (Pipeline einer neuen Art). Du gibst damit auf, dass der erste Plot einer neuen Art im Millisekundenbereich bleibt. Vermutung, nicht gemessen: einige zehn ms für die Treiber-Kompilierung, in einer frischen Session kann die Shader-Kompilierung auf Julia-Seite Sekunden dauern.

---

## 9. `resize!` einer Textur: Texel behalten?

**Problem.** Laut `api:160` erzeugen `resize!(tex, w, h)` und ein Upload neuer Größe ein neues Bild unter einem neuen Index. Was mit dem Inhalt passiert, steht dort nicht. `rr:1304-1305` sagt: "texels kept where the sizes overlap, decision pending". Zum Vergleich:
- Arrays behalten ihren Inhalt bei `resize!` (`api:80`).
- `copy!` mit neuer Länge behält nichts (`api:84`, `:489`).
- Fenstergroße Arrays und Bilder sind nach einem Resize undefiniert (`api:75`, `plan:484-486`).

Die Begründung vom 2026-10-05, der Glyphen-Atlas brauche das, trägt nicht:
- Makies Atlas hält das volle Bild auf der CPU (`data::Matrix{Float16}`, `dev/Makie/Makie/src/utilities/texture_atlas.jl:7`).
- RayMakie lädt bei jeder neuen Glyphe den ganzen Atlas hoch (`dev/Makie/RayMakie/src/overlay/gfx_pipeline.jl:51-62`).
- Makies Atlas wächst heute gar nicht ("Resizing not implemented yet", `texture_atlas.jl:398`, `:424`). Der wachsende Atlas ist eine geplante RayMakie-Änderung (`plan:1695-1699`, `rr:1487-1491`).

Katalog: A19 (`cat-rt:450-454`). Suite: `textures.jl:177` (TX-08), `raymakie_patterns.jl:530` (RM-15 prüft die alten Glyphen-Texel nach dem Resize).

**Optionen.**
- **A. `resize!` behält die Überlappung.**
  - Die Submission, die die Änderung anwendet, kopiert den Bereich min(alt, neu) aus dem alten ins neue Bild, bevor sie das alte ausmustert. Das ist eine Bildkopie pro Resize, im Change-Prefix.
  - Zwei Resizes vor einer Submission werden wie bei Arrays zusammengefasst (das kleinere Behalten gilt).
  - Ein Upload neuer Größe (`copyto!(tex, data)`) schreibt das neue Bild und kopiert nichts, wie `copy!` bei Arrays.
  - TX-08 und RM-15 bleiben.
- **B. Inhalt nach `resize!` undefiniert**, wie bei fenstergroßen Bildern. Es gibt keine Kopie, der Aufrufer lädt neu hoch; RayMakie lädt ohnehin den ganzen Atlas aus Makies CPU-Kopie. TX-08 ändert sich, und RM-15 muss vor der Texel-Prüfung hochladen.

**Empfehlung: A** (wie am 2026-10-05), aber mit anderer Begründung: `resize!` bedeutet dann für Arrays und Texturen dasselbe ("Images are resources like arrays", `plan:1259`). Der Atlas ist kein Grund. Es kostet eine GPU-Kopie pro Textur-Resize, und das kommt selten vor. Messbar aufgeben musst du dafür nichts.

---

## 10. Wert-`GPURef`s und hostwritten-Arrays in Pieces

**Problem.** So kommt ein Wert-`GPURef` heute in einen Kernel:
- `r[] = v` markiert die Pläne in `r.plans` (`int:1414-1426`), die `register!` einträgt (`rr:726`).
- Der Run eines markierten Plans wartet auf seinen vorigen Run und schreibt den Entry (`int:1393-1402`).
- Der Head kopiert den Wert in die Slots.

Ein hostwritten-Graph-Array geht über denselben Entry (`api:144`, `int:1428-1440`).

Ein Piece, das nach dem Compile eingefügt wird (B3.17), hat eine eigene Argument-Region und registriert sich "for its uses only" (`int:832-843`, `:954-965`). `int:867` sagt zwar schon "value GPURef slots included, marked per piece". Aber weder das Entry-Layout (eines pro Plan, beim Compile festgelegt, `plan:877-881`) noch `writeentry!` kennt Pieces. Ein hostwritten-Array, das nur das neue Piece benennt, hat im Plan keine Platzierung.

Katalog: ENTRY-18, Mehrdeutigkeit 21 (`cat-core:208`, `:466`). Suite: `entry.jl:421-452` (ENTRY-18: ein Wert-`GPURef` in einem eingefügten Piece, ein hostwritten-Array in einem anderen, beide erreichen die Pixel, ohne Plan-Kompilierung).

**Optionen.**
- **A** (2026-10-05). Jedes Piece hat eigene Entry-Slots in seiner Argument-Region.
  - `r.plans` enthält auch Pieces. Der Run eines markierten Plans wartet wie heute auf den vorigen Run und schreibt den Wert in den Entry des Plans und in die Slots der markierten Pieces.
  - Ein hostwritten-Array, das nur ein Piece benennt, bekommt seine Bytes in dessen Region, und `x[:] = data` schreibt dorthin.
  - Mantle: `registerpiece!`, `writeentry!` und `x[:] =` in `int` Abschnitt 6/7 ändern sich, und die Piece-Region muss host-sichtbar sein.
  - Es entsteht dieselbe Host-Wartezeit wie heute, nur wenn sich ein Wert geändert hat.
- **B. Draws nehmen gar keine Wert-`GPURef`s oder hostwritten-Arrays**, in keinem Piece, auch nicht im Block von `render!`.
  - Werte pro Frame sind in Draws Ein-Element-Device-Arrays, die per Store geschrieben werden und nie warten. `plan:1210-1215` empfiehlt das ohnehin.
  - Das ist am einfachsten, und RayMakie benutzt nur Arrays (`rr:1431-1447`).
  - Generischer Draw-Code, der ein `GPURef` übergibt, wirft bei der Deklaration.
- **C. Liest ein eingefügtes Piece ein Wert-`GPURef` oder hostwritten-Array, das der Entry des Plans nicht hält, wird der Plan zur Neukompilierung markiert** (der Pfad von B3.16). Das ist korrekt und einfach, aber solche Inserts kompilieren neu, gegen B3.17.

**Empfehlung: A** (wie am 2026-10-05). Nur mit A verhält sich ein `draw!` im Block von `render!` (mit dem Plan kompiliert) und in einem eingefügten Piece gleich. B verbietet Draws, was Kernel dürfen, und C bricht "ein Insert kompiliert nie neu". Du gibst damit eine Verwaltung pro Piece auf, die für RayMakie allein nicht nötig wäre; dort würde B reichen.

---

## 11. Reihenfolge der Pieces

**Problem.** `api:176` sagt: "a changed order is a list rewrite". `plan:1250-1251` und `plan:1689-1691` sagen dasselbe. Einen Aufruf dafür nennt aber keine der Stellen (Mehrdeutigkeit 23, `cat-core:468`; Ledger F14, `refactor-ledger.md:961`; offene Entscheidung 15, `plan:2326-2330`). `insert!(p)` hängt nur hinten an (`int:917`).

In Makie bestimmt die z-Translation, in welcher Reihenfolge 2D-Plots gezeichnet werden. GLMakie sortiert seine Render-Liste in jedem Frame mit `sort!(screen.renderlist; by = …)` (`dev/Makie/GLMakie/src/postprocessing.jl:139-152`). RayMakie muss einen Plot vor einen anderen ziehen können, ohne ihn neu einzufügen. Es gibt keinen Suite-Fall dafür.

**Optionen.**
- **A** (2026-10-05). `reorder!(p, handles)` (Name offen; nicht in Base und nicht in Mantle): die vollständige neue Reihenfolge als Vektor von Handles. Der nächste Run schreibt die Liste neu; ein Teil wird neu aufgenommen, nichts kompiliert.
- **B. `Base.sort!(p; by, lt, rev)` für den Pass überladen.** Die Pieces werden nach einem Schlüssel sortiert, den der Aufrufer pro Handle berechnet. Der Mechanismus ist derselbe wie bei A.
  - RayMakie schreibt `sort!(screen.pass; by = h -> zvalue(h))`, wie GLMakie.
  - Julias Standard-Sortierung ist für einen Vektor von Objekten stabil (`Base.Sort.defalg(Any[1, 2]) == DefaultStable()` auf 1.12.7, geprüft), also behalten Pieces mit gleichem z die Einfügereihenfolge.
  - Ein Aufruf, der die Reihenfolge nicht ändert, vermerkt nichts.
- **C. Positionsangabe beim Einfügen**: `insert!(p; before = h) do … end`. Verschieben heißt `delete!` und `insert!` dieses Plots. Das Piece wird dabei neu kompiliert und neu aufgenommen (Pipelines aus dem Cache); das dauert pro verschobenem Plot im Millisekundenbereich, die Grenze aus B3.17.
- **D. Stand belassen**: nur Anhängen. Um einen Plot zu verschieben, löschst du ihn und jeden Plot dahinter und fügst sie neu ein.

Das Sortieren nach Transparenz in jedem Frame bleibt in jeder Option offene Entscheidung 15.

**Empfehlung: B.** Das ist eine Änderung gegenüber dem 2026-10-05. Die Wirkung ist dieselbe wie bei `reorder!`, aber es braucht keinen neuen Funktionsnamen (deine Regel: Multiple Dispatch vor neuen Namen), und es ist genau der Aufruf, den GLMakie schon macht. Du gibst damit auf, dass beliebige Reihenfolgen direkt gehen; sie müssen als Schlüssel formuliert werden (`by = h -> position[h]`).

---

## 12. `push!` über `maxInstanceCount`

**Problem.** Vulkan begrenzt die Instanzzahl eines Builds und der Größenabfrage (VUID-03801 und VUID-03785, `cat-rt:430`). `maxInstanceCount` ist auf allen vier Vulkan-Geräten dieser Maschine 16 777 215 (2^24 − 1): RADV RX 7900 XTX, RADV Raphael, NVIDIA RTX 4000 Ada, llvmpipe (`vulkaninfo`, 2026-10-06). `rr:1058` wirft bereits in `push!` ("a total past the device's maxInstanceCount throws here").

Nicht abgedeckt ist dreierlei:
1. Die Kapazität verdoppelt sich ab 64 (`api:151`, `rr:1119-1121`). Sie erreicht 2^24 = 16 777 216 und damit mehr als das Limit, sobald die Gesamtzahl 2^23 = 8 388 608 übersteigt. Speicher und Größenabfrage überschreiten das Limit dann, obwohl die Gesamtzahl darunter liegt.
2. Eine vom Kernel geschriebene Gesamtzahl wird über die Kapazität gebaut (`rr:959`, `:963-968`).
3. Die Gesamtzahl wächst nicht nur durch `push!`, sondern auch durch `resize!` und `copy!` einer Quelle (`rr:1111-1117`) und durch das Umbinden einer Array-`GPURef`-Quelle auf ein längeres Array (`rr:1251-1252`).

Es betrifft ray-getraceten Meshscatter mit vielen Partikeln (`rr:1498-1514`) und Hikari-Szenen. Katalog: TL-18, A12 (`cat-rt:270`, `:430`). Suite: `tlas.jl:433-446` (TL-18: Fehler bei `push!`, nichts ausstehend, keine Größenabfrage, Kapazität ≤ Limit).

**Optionen.**
- **A. Fehler bei dem Aufruf, der die angeforderte Gesamtzahl erhöht**: `push!`, außerdem `resize!`, `copy!` und Umbinden einer Quelle.
  - Diese Aufrufe sperren das Array schon zusammen mit seinen Lesern und vermerken den TLAS-Build (`rr:1111-1117`). Die Prüfung ist ein Vergleich mit der Gesamtzahl, die der TLAS auf dem Host führt.
  - Bei einer vom Kernel geschriebenen Quelle zählt deren Kapazität.
  - Die Kapazität wird beim Limit gekappt.
  - Der Fehler kommt in der Zeile, die ihn verursacht hat, und es bleibt nichts ausstehend.
- **B. Fehler nur in `push!`** (wie `rr:1058` heute), plus Kappen der Kapazität. Ein `resize!` einer Quelle über das Limit fällt erst der Submission auf, die den Build anwendet. Die wirft dann aus `run!` oder einem eager Aufruf, weit weg von der Ursache.
- **C. Kappen**: höchstens `maxInstanceCount` bauen, Instanzen darüber werden nie getroffen, einmal eine Warnung.

Das Kappen der Kapazität brauchen alle drei Optionen, sonst ist schon eine Szene mit 8.4 Mio. Instanzen ungültig.

**Empfehlung: A.** Die Regel ist dieselbe wie am 2026-10-05 ("wirft beim Aufruf"), aber jetzt für alle Aufrufe, die die Gesamtzahl ändern, und mit gekappter Kapazität. Ohne das Kappen scheitert TL-18s `t.capacity <= 1000`, und eine Gesamtzahl zwischen 2^23 und dem Limit bricht. Du gibst damit auf: Szenen über 16.7 Mio. Instanzen in einem TLAS. Die waren vorher schon ungültig, nur fiel es nicht auf.

---

## 13. Mantle lädt nicht ohne Display

**Problem.** `using Mantle` und Mantles Vorkompilierung scheitern ohne Display mit `InitError: GLFWError(65550, "X11: The DISPLAY environment variable is missing"), glfwInit failed`. Die Ursache:
- GLFW ist eine harte Abhängigkeit von Mantle (`dev/Mantle/Project.toml:30`).
- Benutzt wird es nur für Fenster:
  - `src/vulkan/graphics/window.jl` (Import in `:6`, `GLFW.Init()` beim Anlegen eines Fensters in `:79`),
  - `src/metal/metal.jl:40` und `src/metal/window.jl`,
  - `src/vulkan/graph.jl:1644` (`GLFW.PollEvents()`).
- GLFW.jl 3.4.6 ruft in `__init__` schon beim Laden `Init()` auf und wirft, wenn das scheitert (`~/.julia/packages/GLFW/wA4ue/src/GLFW.jl:50-71`, `throw(errors)` in `:69`).

Das Anlegen des Devices selbst braucht GLFW nicht: Die Vulkan-Instanz nimmt die Surface-Extensions, die der Loader anbietet (`src/vulkan/runtime/device.jl:1085-1108`). Trotzdem können ein Server mit einem DNN-Modell, CI ohne X und jede bt-Session ohne gesetztes `DISPLAY`/`XAUTHORITY` Mantle gar nicht laden, obwohl nur Fenster GLFW brauchen. `api` 2.7 (`api:186-192`) und 4.7 (`api:349-356`, G13 `api:604-606`) beschreiben Fenster als Kern-API.

**Optionen.**
- **A. GLFW.jl initialisiert lazy.** `__init__` warnt statt zu werfen, und der erste GLFW-Aufruf, der die Initialisierung braucht (das Anlegen eines Fensters), wirft.
  - Umsetzung: eine Upstream-Änderung in JuliaGL/GLFW.jl, das du pflegst. Ein paar Zeilen in `__init__`, und `atexit(Terminate)` erst nach dem ersten erfolgreichen `Init()`.
  - Mantle bleibt unverändert; es ruft beim Anlegen eines Fensters schon `GLFW.Init()` auf (`window.jl:79`).
  - Es behebt auch das Laden von GLMakie ohne Display. Das ist wichtig, weil der Editor von GLMakie abhängt (`dev/VideoEditor/Project.toml:23`) und RayMakie von GLFW (`dev/Makie/RayMakie/Project.toml:11`).
  - Braucht ein GLFW.jl-Release. Bis dahin ändert sich nichts.
  - Risiko: Code, der GLFW-Funktionen ohne erfolgreiche Initialisierung aufruft, bekommt den Fehler bei diesem Aufruf statt beim Laden. Vermutung: Das Anlegen eines GLMakie-Screens ist so ein Aufruf und scheitert dort mit demselben GLFW-Fehler.
- **B. Mantle verschiebt den GLFW-Code in eine Paket-Extension** (`MantleGLFWExt`, ausgelöst durch GLFW; Metals Fenster braucht zusätzlich ObjectiveC, `src/metal/metal.jl:36-40`).
  - Ohne `using GLFW` ist `Window(dev, …)` ein `MethodError`. Einen Fallback gibt es nicht (`api:43-44`).
  - `api` 2.7 und 4.7 ändern sich: Fenster gibt es nach `using GLFW` oder nach einem Paket, das GLFW lädt (RayMakie, GLMakie).
  - DNN-Runner und CI laden Mantle ohne Display.
  - RayMakie und der Editor laden GLFW aber selbst und scheitern ohne Display weiterhin, etwa beim Offscreen-Raytracing auf einem Server, solange A nicht auch passiert.
  - Mantles Vorkompilier-Workloads, die Fenster anfassen, wandern in die Extension.
- **C. Belassen und dokumentieren**: Ein Display ist Pflicht. CI läuft unter `xvfb-run`, Server brauchen einen virtuellen X-Server, und jede bt-Session setzt `DISPLAY`/`XAUTHORITY` vor dem Laden.

Folgen je Nutzer:

| Nutzer | A | B | C |
|---|---|---|---|
| DNN-Runner (Server) | laden ohne Display | laden ohne Display | brauchen Xvfb |
| RayMakie | lädt ohne Display; Fenster brauchen weiter eins, Offscreen-Rendern in `Image`s geht | unverändert: scheitert ohne Display | unverändert |
| Video-Editor (über GLMakie) | lädt ohne Display | unverändert | unverändert |
| CI | Suite ohne Display; `:window`-Fälle übersprungen oder unter Xvfb | Mantle-Suite ohne Display, RayMakie-Fälle nicht | alles unter Xvfb |
| bt-Sessions | `DISPLAY` nur nötig, wenn ein Fenster geöffnet wird | nur für Sessions ohne RayMakie/GLMakie | immer |

**Empfehlung: A**, mit C als Doku bis zum GLFW.jl-Release. A ist die einzige Option, die auch RayMakie, GLMakie und den Editor ohne Display ladbar macht, und sie ändert keine Mantle-API. B kann später dazukommen, falls das Laden der GLFW-Bibliothek messbar Ladezeit kostet (nicht gemessen); allein behebt B nur reine Compute-Nutzer. Du gibst damit auf: den Fehler beim Laden. Wer ohne Display ein Fenster öffnet, bekommt ihn erst dort.
