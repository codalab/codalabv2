<comp-phase-info>

    <div class="ui container grid">
        <div class="row">
            <div class="sixteen wide column">
                <div align="center" class="ui stacked centered message">
                    <h5>Competition Run Time</h5>
                    <div class="ui ordered centered steps">
                        <div each="{competition.phases}" class="step">
                            <div class="content">
                                <div class="title">{title}</div>
                                <div class="description">{description}</div>
                            </div>
                        </div>
                        <!--<div class="completed step">-->
                            <!--<div class="content">-->
                                <!--<div class="title">Phase 2: Validation Test</div>-->
                                <!--<div class="description">Enter billing information</div>-->
                            <!--</div>-->
                        <!--</div>-->
                        <!--<div class="active step">-->
                            <!--<div class="content">-->
                                <!--<div class="title">Phase 3: Feedback Final</div>-->
                                <!--<div class="description">Verify order details</div>-->
                            <!--</div>-->
                        <!--</div>-->
                        <!--<div class="step">-->
                            <!--<div class="content">-->
                                <!--<div class="title">Phase 4: Blind Final</div>-->
                                <!--<div class="description">Verify order details</div>-->
                            <!--</div>-->
                        <!--</div>-->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        var self = this
        self.competition = opts.competition
    </script>
</comp-phase-info>