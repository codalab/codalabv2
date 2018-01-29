<competition-details>
    <div class="ui form">
        <div class="field required">
            <label>Title</label>
            <input ref="title">
        </div>

        <!--<div class="field required">
            <label>Logo</label>
            <input type="file">
        </div>-->


        <div class="field required">
            <label>Logo</label>

            <input-file name="logo" accept="image/*"></input-file>
        </div>

        <!--<div class="two fields">
            <div class="ui calendar field required" ref="calendar_start">
                <label>Start</label>
                <div class="ui input left icon">
                    <i class="calendar icon"></i>
                    <input type="text" ref="start">
                </div>
            </div>

            <div class="ui calendar field" ref="calendar_end">
                <label>End</label>
                <div class="ui input left icon">
                    <i class="calendar icon"></i>
                    <input type="text" ref="end">
                </div>
            </div>
        </div>-->

        <!--<div class="field required">
            <label>Description</label>
            <p>Uses <a href="https://simplemde.com/markdown-guide">markdown</a> formatting</p>
            <textarea class="markdown-editor" ref="description"></textarea>
        </div>-->
        <!--                        <div class="field"> -->
        <!--                            <label>Short Text</label> -->
        <!--                            <textarea rows="2"></textarea> -->
        <!--                        </div> -->
    </div>

    <script>
        var self = this

        /*---------------------------------------------------------------------
         Init
        ---------------------------------------------------------------------*/
        self.data = {}

        // We temporarily store this to display it nicely to the user, could be a behavior we break out into its own
        // component later!
        self.logo_file_name = ''

        self.one("mount", function () {
            /*
            // datetime pickers
            var datetime_options = {
                type: 'date',
                popupOptions: {
                    position: 'bottom left',
                    lastResort: 'bottom left',
                    hideOnScroll: false
                },
                onHide: function () {
                    // Have to do this because onchange isn't fired when date is picked
                    self.form_update()
                }
            }
            var start_options = Object.assign({}, datetime_options, {endCalendar: self.refs.calendar_end})
            var end_options = Object.assign({}, datetime_options, {startCalendar: self.refs.calendar_start})

            $(self.refs.calendar_start).calendar(start_options)
            $(self.refs.calendar_end).calendar(end_options)

            // awesome markdown editor
            $(self.refs.description).each(function (i, ele) {
                new SimpleMDE({element: ele})
            })
            */

            // logo selection
            /*$(self.refs.logo).on('change', function (event) {
                // Value comes like c:/fakepath/file_name.txt -- cut out everything but file_name.txt
                self.logo_file_name = self.refs.logo.value.replace(/\\/g, '/').replace(/.*\//, '')
                self.update()
            })*/

            // Form change events
            $(':input', self.root).not('[type="file"]').not('button').not('[readonly]').each(function (i, field) {
                this.addEventListener('keydown', self.form_update)
            })

            // Capture and convert logo to base64 for easy uploading
            $('input[name="logo"]', self.root).change(function() {
                getBase64(this.files[0]).then(function(data) {
                    self.data['logo'] = data
                    self.form_update()
                })
            })
        })

        /*---------------------------------------------------------------------
         Methods
        ---------------------------------------------------------------------*/
        self.form_update = function () {
            var is_valid = true

            // NOTE: logo is excluded here because it is converted to 64 upon changing and set that way
            self.data['title'] = self.refs.title.value

            if(!self.data['title'] || !self.data['logo']) {
                is_valid = false
            }

            CODALAB.events.trigger('competition_is_valid_update', 'details', is_valid)

            if(is_valid) {
                CODALAB.events.trigger('competition_data_update', self.data)
            }
        }
    </script>
</competition-details>