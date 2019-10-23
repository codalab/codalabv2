<queues-list>
    <div class="ui icon input">
        <input type="text" placeholder="Search by name..." ref="search" onkeyup="{filter.bind(this, undefined)}">
        <i class="search icon"></i>
    </div>
    <div class="ui checkbox" onclick="{ filter.bind(this, undefined) }">
        <label>Show Public Queues</label>
        <input type="checkbox" ref="public">
    </div>
    <div class="ui green right floated labeled icon button" onclick="{ show_modal.bind(this, undefined) }"><i class="add circle icon"></i>
        Create Queue
    </div>
    <table class="ui {selectable: queues.length > 0} celled compact table">
        <thead>
        <tr>
            <th>Name</th>
            <th width="150px">Owner</th>
            <th width="125px">Created</th>
            <th width="50px">Public</th>
            <th width="150px">Actions</th>
        </tr>
        </thead>
        <tbody>
        <tr each="{ queue in queues }" class="queue-row">
            <td>{ queue.name }</td>
            <td>{ queue.owner }</td>
            <td>{ timeSince(Date.parse(queue.created_when)) } ago</td>
            <td class="center aligned">
                <i class="checkmark box icon green" if="{ queue.is_public }"></i>
            </td>
            <td class="center aligned">
                <span class="popup" data-tooltip="{queue.broker_url}" data-position="top center">
                    <i class="icon eye popup-button" if="{ !!queue.broker_url }"></i>
                </span>
                <span class="popup" data-tooltip="Copy Broker URL">
                    <i class="icon copy popup-button" if="{ !!queue.broker_url }"
                       onclick="{ copy_queue_url.bind(this, queue) }"></i>
                </span>
                <span class="popup" data-tooltip="Edit Queue">
                    <i class="blue icon edit popup-button" if="{ queue.is_owner && !!queue.broker_url }"
                       <!--onclick="{ edit_queue.bind(this, queue) }"></i>-->
                       onclick="{ show_modal.bind(this, queue) }"></i>
                </span>
                <span class="popup" data-tooltip="Delete Queue">
                    <i class="red icon trash popup-button" if="{ queue.is_owner && !!queue.broker_url }"
                       onclick="{ delete_queue.bind(this, queue) }"></i>
                </span>
            </td>
        </tr>

        <tr if="{queues.length === 0}">
            <td class="center aligned" colspan="4">
                <em>No Queues Yet!</em>
            </td>
        </tr>
        </tbody>
        <tfoot>
        <!-------------------------------------
                  Pagination
        ------------------------------------->
        <tr if="{queues.length > 0 && _.get(pagination, 'next')}">
            <th colspan="6">
                <div class="ui right floated pagination menu" if="{queues.length > 0}">
                    <a show="{!!_.get(pagination, 'previous')}" class="icon item" onclick="{previous_page}">
                        <i class="left chevron icon"></i>
                    </a>
                    <div class="item">
                        <label>{page}</label>
                    </div>
                    <a show="{!!_.get(pagination, 'next')}" class="icon item" onclick="{next_page}">
                        <i class="right chevron icon"></i>
                    </a>
                </div>
            </th>
        </tr>
        </tfoot>
    </table>

    <div class="ui modal" ref="modal">
        <div class="header">
            Queue Form
        </div>
        <div class="content">
            <form class="ui form" ref="form">
                <div class="ui active tab" data-tab="details">
                    <div class="required field">
                        <label>Name</label>
                        <input name="name" placeholder="Name" ref="queue_name" value="{ _.get(selected_queue, 'name', '') }">
                    </div>
                    <div class="field">
                        <div class="ui checkbox">
                            <label>Is Public?</label>
                            <input type="checkbox" ref="queue_public">
                        </div>
                    </div>
                    <div class="field">
                        <label>Collaborators</label>
                        <div class="ui fluid search multiple selection dropdown collab-search">
                            <input type="hidden" name="collaborators" ref="queue_collaborators">
                            <i class="dropdown icon"></i>
                            <div class="default text">Select Collaborator</div>
                            <div class="menu">
                            </div>
                        </div>
                    </div>
                </div>
            </form>
        </div>
        <div class="actions">
            <div class="ui primary button" onclick="{ handle_queue }">Submit</div>
            <div class="ui basic red cancel button" onclick="{ close_modal }">Cancel</div>
        </div>
    </div>

    <script>
        var self = this
        self.queues = []
        self.selected_queue = {}
        self.page = 1

        self.one("mount", function () {
            self.update_queues()
            $(".ui.checkbox", self.root).checkbox()
            $('.collab-search')
                .dropdown({
                    apiSettings: {
                        url: `${URLS.API}user_lookup/?q={query}`,
                    },
                    clearable: true,
                    preserveHTML: false,
                    //minCharacters: 2,
                    fields: {
                        remoteValues: 'results',
                        name: 'username',
                        value: 'id',
                    },
                    cache: false,
                    maxResults: 5,
                })
            ;
        })

        self.switch_to_form = () => {
            window.location = '/queues/form/'
        }

        self.update_queues = function (filters) {
            filters = filters || {}
            let show_public_queues = $(self.refs.public).prop('checked')
            if (show_public_queues) {
                filters.public = true
            }
            CODALAB.api.get_queues(filters)
                .done(function (data) {
                    self.queues = data.results
                    self.pagination = {
                        "count": data.count,
                        "next": data.next,
                        "previous": data.previous
                    }
                    self.update()
                })
                .fail(function (response) {
                    toastr.error("Could not load tasks")
                })
        }

        /*---------------------------------------------------------------------
        Modal Methods
        ----------------------------------------------------------------------*/
        self.show_modal = (queue) => {
            if (queue !== undefined && queue !== null) {
                self.set_selected_queue(queue)
            }
            $(self.refs.modal).modal('show')
        }

        self.close_modal = () => {
            $(self.refs.modal).modal('hide')
            self.clear_form()
        }

        self.clear_form = () => {
            $('.collab-search').dropdown('clear')
            self.refs.queue_name.value = ''
            self.selected_queue = {}
            self.refs.queue_public.checked = false
        }

        self.set_selected_queue = function (queue) {
            self.selected_queue = queue
            $('.collab-search').dropdown('setup menu', {
                values: _.values(_.mapValues(queue.organizers, function (organizer) {
                    return {id: organizer.id, username: organizer.username}
                }))
            })
            $('.collab-search').dropdown('set selected',  _.values(_.mapValues(queue.organizers, function (organizer) {
                return organizer.username
            })))
            if (queue.is_public) {
                self.refs.queue_public.checked = true
            }
        }

        self.handle_queue = function () {
            var data = {
                name: self.refs.queue_name.value,
                is_public: self.refs.queue_public.checked,
            }
            var endpoint
            // TODO: Better way to handle collaborators?
            if ($('.collab-search').dropdown('get value') != ''){
                data.organizers = $('.collab-search').dropdown('get value').split(",")
            } else {
                data.organizers = []
            }
            if (!_.isEmpty(self.selected_queue)) {
                endpoint = CODALAB.api.update_queue(self.selected_queue.id, data)
            } else {
                endpoint = CODALAB.api.create_queue(data)
            }
            endpoint
                .done(function (response) {
                    toastr.success("Succesfully updated/made queue!")
                    self.close_modal()
                    self.update_queues()
                })
                .fail(function (response) {
                    toastr.error("Could not update/create queue!")
                })
        }

        /*---------------------------------------------------------------------
         Table Methods
         ---------------------------------------------------------------------*/
        self.filter = function (filters) {
            filters = filters || {}
            _.defaults(filters, {
                search: $(self.refs.search).val(),
                page: 1,
            })
            self.page = filters.page
            self.update_queues(filters)
        }

        self.next_page = function () {
            if (!!self.pagination.next) {
                self.page += 1
                self.filter({page: self.page})
            } else {
                alert("No valid page to go to!")
            }
        }
        self.previous_page = function () {
            if (!!self.pagination.previous) {
                self.page -= 1
                self.filter({page: self.page})
            } else {
                alert("No valid page to go to!")
            }
        }

        self.delete_queue = function (queue) {
            if (confirm("Are you sure you want to delete '" + queue.name + "'?")) {
                CODALAB.api.delete_queue(queue.id)
                    .done(function () {
                        self.update_queues()
                        toastr.success("Queue deleted successfully!")
                    })
                    .fail(function () {
                        toastr.error("Could not delete queue!")
                    })
            }
            event.stopPropagation()
        }

        self.copy_queue_url = function (queue) {
            navigator.clipboard.writeText(queue.broker_url).then(function () {
                /* clipboard successfully set */
                toastr.success("Successfully copied broker url to clipboard!")
            }, function () {
                /* clipboard write failed */
                toastr.error("Failed to copy broker url to clipboard!")
            });
        }

        self.edit_queue = function (queue) {
            window.location = '/queues/form/' + queue.id
            event.stopPropagation()
        }
    </script>
    <style>
        .popup-button {
            cursor: pointer;
        }
    </style>
</queues-list>
