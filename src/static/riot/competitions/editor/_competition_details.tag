<competition-details>
    <div class="ui form">
        <div class="field required">
            <label>Title</label>
            <input ref="title">
        </div>

        <div class="field required">
            <label>Logo</label>

            <!-- This is the SINGLE FILE with NO OTHER OPTIONS example -->
            <!-- In the future, we'll have this type AND a type that is pre-filled with nice options -->
            <label show="{ uploaded_logo }">Uploaded Logo: <a href="{ uploaded_logo }">{ uploaded_logo_name }</a></label>
            <div class="ui left action file input">
                <button class="ui icon button" onclick="document.getElementById('form_file_logo').click()">
                    <i class="attach icon"></i>
                </button>
                <input id="form_file_logo" type="file" ref="logo" accept="image/*">

                <!-- Just showing the file after it is uploaded -->
                <input value="{ logo_file_name }" readonly onclick="document.getElementById('form_file_logo').click()">
            </div>
        </div>
        <div class="field smaller-mde">
            <label>Description</label>
            <textarea class="markdown-editor" ref="comp_description" name="description"></textarea>
        </div>
    </div>

    <script>
        var self = this

        /*---------------------------------------------------------------------
         Init
        ---------------------------------------------------------------------*/
        self.data = {}
        self.is_editing_competition = false

        // We temporarily store this to display it nicely to the user, could be a behavior we break out into its own
        // component later!
        self.logo_file_name = ''

        self.one("mount", function () {
            self.markdown_editor = new EasyMDE({
                element: self.refs.comp_description,
                autoRefresh: true,
                forceSync: true,
                hideIcons: ["preview", "side-by-side", "fullscreen"]
            })
            // Form change events
            $(':input', self.root).not('[type="file"]').not('button').not('[readonly]').each(function (i, field) {
                this.addEventListener('keyup', self.form_updated)
            })

            // Capture and convert logo to base64 for easy uploading
            $(self.refs.logo).change(function() {
                self.logo_file_name = self.refs.logo.value.replace(/\\/g, '/').replace(/.*\//, '')
                self.update()
                getBase64(this.files[0]).then(function(data) {
                    self.data['logo'] = JSON.stringify({file_name: self.logo_file_name, data: data})
                    self.form_updated()
                })
            })
        })

        /*---------------------------------------------------------------------
         Methods
        ---------------------------------------------------------------------*/
        self.form_updated = function () {
            var is_valid = true

            // NOTE: logo is excluded here because it is converted to 64 upon changing and set that way
            self.data['title'] = self.refs.title.value
            self.data['description'] = self.markdown_editor.value()

            // Require title, logo is optional IF we are editing -- will just keep the old one if
            // a new one is not provided
            if(!self.data['title'] || (!self.data['logo'] && !self.is_editing_competition)) {
                is_valid = false
            }

            CODALAB.events.trigger('competition_is_valid_update', 'details', is_valid)

            if(is_valid) {
                // If we don't have logo data AND we're editing, put in empty data
                if(!self.data['logo'] && self.is_editing_competition){
                    self.data['logo'] = undefined
                }
                CODALAB.events.trigger('competition_data_update', self.data)
            }
        }

        /*---------------------------------------------------------------------
         Events
        ---------------------------------------------------------------------*/
        CODALAB.events.on('competition_loaded', function(competition){
            self.is_editing_competition = true

            self.refs.title.value = competition.title
            self.markdown_editor.value(competition.description || '')

            // Value comes like c:/fakepath/file_name.txt -- cut out everything but file_name.txt
            self.uploaded_logo_name = competition.logo.replace(/\\/g, '/').replace(/.*\//, '')
            self.uploaded_logo = competition.logo
            self.form_updated()
        })
    </script>
</competition-details>