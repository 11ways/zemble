class AppDirectory {
    private static Fix fixOf(Panel panel, Record record, Verdict verdict) {
        if (verdict.fixes().isEmpty()) {
            return null;
        }
        Offered offered = Bands.offeredFixes(panel, record, verdict.fixes());
        for (Identifier fix : verdict.fixes()) {
            for (LinkState link : offered.inlineLinks()) {
                if (link.id().equals(fix)) {
                    return Fix.ofLink(link);
                }
            }
            for (InvokeState invoke : offered.inlineInvokes()) {
                if (invoke.id().equals(fix) && invoke.disabledReason() == null) {
                    return Fix.ofInvoke(invoke);
                }
            }
        }
        return null;
    }
}
