class Caller {
    private static Fix firstFix(Panel panel, Record record, Verdict verdict) {
        if (verdict.fixes().isEmpty()) {
            return null;
        }
        Fix found = Health.fixCell(panel, record, verdict);
        for (Identifier fix : verdict.fixes()) {
            for (LinkState link : Bands.offeredFixes(panel, record, verdict.fixes()).inlineLinks()) {
                if (link.id().equals(fix)) {
                    return found;
                }
            }
        }
        return null;
    }
}
