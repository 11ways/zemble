# Architecture

Some prose that mentions zenit but is not a table.

| Capability | Mechanism home | Consumers (thin wiring) |
| --- | --- | --- |
| Pagination arithmetic (`PageWindow.of(page, total, size)` -> offset) | `zenit` core (`common/data`) | `RecordSourceHandlers`, zenit-cms `Paging` |
| Secure tokens (`SecureTokens`: sha256 hex, LINE\|BYTES units) | `zenit` core (`server/security`) | zenit-auth (tokens), hohenheim (signatures) |
| Form rendering primitives | `zenit-forms` + `plumage` (UI components) | zenit-cms templates |
| Broken row with too few cells |
| Charts and sparklines | `SomethingUndeclared` (not a module) | apps |

| Other table | Notes |
| --- | --- |
| ignored | because its headers do not match |
