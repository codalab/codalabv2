<registration>
    <div if="{!status}" class="ui grid">
        <div class="row">
            <p>You have not yet registered for this competition.</p>
            <p>
                To participate in this competition, you must accept its specific <a href="" onclick="{show_modal}">terms and conditions</a>. After you
                register, the competition organizer will review your application and notify you when your participation
                is approved.
            </p>
            <p if="!registration_auto_approve">
                This competition <strong>requires approval</strong> from the competition organizers. After submitting your registration request, an email
                will be sent to the competition organizers notifying them of your request. Your application will remain pending until they
                approve or deny it.
            </p>
        </div>
        <div class="row">
            <div class="ui checkbox">
                <input type="checkbox" id="accept-terms" onclick="{accept_toggle}">
                <label for="accept-terms">I accept the terms and conditions of the competition.</label>
            </div>
        </div>
        <div class="row">
            <button class="ui primary button {disabled: !accepted}" onclick="{submit_registration}">
                Register
            </button>
        </div>
    </div>

    <div if="{status}">
        Your current status is: {_.startCase(status)}
    </div>

    <div ref="terms_modal" class="ui modal">
        <div class="header">
            Terms and Conditions
        </div>
        <div ref="terms_content" class="content">

        </div>
        <div class="actions">
            <div class="ui cancel button">
                Close
            </div>
        </div>
    </div>

    <script>
        let self = this
        self.on('mount', () => {
            self.accepted = false
        })

        CODALAB.events.on('competition_loaded', (competition) => {
            self.competition_id = competition.id
            if (self.refs.terms_content) {
                self.refs.terms_content.innerHTML = render_markdown(competition.terms)
            }
            self.registration_auto_approve = competition.registration_auto_approve
            self.status = competition.participant_status
            self.update()
        })

        self.accept_toggle = () => {
            self.accepted = !self.accepted
        }

        self.show_modal = (e) => {
            if (e) {
                e.preventDefault()
            }
            console.log('hello')
            $(self.refs.terms_modal).modal('show')
        }

        self.submit_registration = () => {
            CODALAB.api.submit_competition_registration(self.competition_id)
                .done(response => {
                    self.status = response.participant_status
                    if (self.status === 'approved') {
                        toastr.success('You have been registered!')
                        CODALAB.api.get_competition(self.competition_id)
                            .done(competition => {
                                CODALAB.events.trigger('competition_loaded', competition)
                            })
                    } else {
                        toastr.success('Your registration application is being processed!')
                    }
                    self.update()
                })
                .fail(response => {
                    toastr.error('Something went wrong.')
                })
        }
    </script>
</registration>