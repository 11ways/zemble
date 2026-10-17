"""nib provider mode: tool answers that say where code is, for nib to print from the live file.

The protocol is nib's (`docs/nib-provider.md`): every answer carries `structuredContent`
`{"nib": 1, "parts": [...]}` beside its unchanged text, and code only ever travels as a location.
"""
