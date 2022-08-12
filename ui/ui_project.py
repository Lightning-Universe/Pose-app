import numpy as np
import streamlit as st


def run():

    # ----------------------------------------------------
    # landing
    # ----------------------------------------------------

    st.markdown(
        """
        ## Manage Lightning Pose project
        """
    )

    st_mode = st.radio(
        "Create new project or load existing?",
        options=["Create new project", "Load existing project"]
    )

    st_project_name = st.text_input("Enter project name", value="")

    if st_project_name and st_mode == "Load existing project":
        project_loaded = st.button(
            "Load project", disabled=True if not st_project_name != "" else False)
    else:
        project_loaded = False

    if st_project_name:
        if st_mode == "Load existing project" and project_loaded:
            enter_data = True
        elif st_mode == "Create new project" and st_project_name != "":
            enter_data = True
        else:
            enter_data = False
    else:
        enter_data = False

    # add some whitespace
    st.markdown("")
    st.markdown("")

    # ----------------------------------------------------
    # user input for data config
    # ----------------------------------------------------

    st_n_views = 0
    st_keypoints = []
    st_n_keypoints = 0
    st_pcasv_columns = []
    st_pcamv_columns = np.array([])

    if enter_data:
        st.markdown("##### Camera views")
        n_views = st.text_input("Enter number of camera views:", disabled=not enter_data)
        if n_views:
            st_n_views = int(n_views)
        else:
            st_n_views = 0
        st.markdown("")

    if st_n_views > 0:
        st.markdown("##### Define keypoints")
        keypoint_instructions = """
            **Instructions**: 
            If your data has multiple views, make sure to create an entry for each bodypart
            in each view below like in the following example with 2 views (top and bottom): 
            ```
            nose_top
            l_ear_top
            r_ear_top
            nose_bottom
            l_ear_bottom
            r_ear_bottom
            corner1_top
            ```
            It is also possible to track keypoints that are only present in a subset of the views,
            such as the keypoint `corner1_top` above.
            
            The order in which you list the keypoints here determines the labeling order.
        """
        st.markdown(keypoint_instructions)
        keypoints = st.text_area("Enter keypoint names (one per line):", disabled=not enter_data)
        st_keypoints = keypoints.strip().split('\n')
        if len(st_keypoints) == 1 and st_keypoints[0] == "":
            st_keypoints = []
        st_n_keypoints = len(st_keypoints)
        st.markdown(f"You have defined {st_n_keypoints} keypoints across {st_n_views} views")
        st.markdown("")

    if len(st_keypoints) > 1:
        st.markdown("##### Select subset of keypoints for Pose PCA")
        st.markdown("""
            **Instructions**:
            The selected subset will be used for a Pose PCA loss on unlabeled videos. 
            The subset should be keypoints that are not usually occluded (such as a tongue) 
            and are not static (such as the corner of a box).
        """)
        pcasv_selected = [False for _ in st_keypoints]
        for k, kp in enumerate(st_keypoints):
            pcasv_selected[k] = st.checkbox(kp, disabled=not enter_data)
        st_pcasv_columns = list(np.where(pcasv_selected)[0])
        st.markdown("")

    if len(st_keypoints) > 1 and st_n_views > 1:

        st.markdown("##### Select subset of body parts for Multiview PCA")
        st.markdown("""
            **Instructions**:
            The selected subset will be used for a Multiview PCA loss on unlabeled videos. 
            The subset should be keypoints that are usually visible in all camera views.
        """)
        # pcasv_selected = [False for _ in st_keypoints]
        # for k, kp in enumerate(st_keypoints):
        #     pcasv_selected[k] = st.checkbox(kp, disabled=not enter_data)
        # st.markdown("")
        n_bodyparts = st.text_input("Enter number of body parts visible in all views:")
        if n_bodyparts:
            st_n_bodyparts = int(n_bodyparts)
        else:
            st_n_bodyparts = 0

        if st_n_bodyparts > 0:

            st_pcamv_columns = np.zeros((st_n_views, st_n_bodyparts), dtype=np.int)

            # set column titles
            cols_title = st.columns(st_n_views + 1)
            for c, col in enumerate(cols_title[1:]):
                col.text(f"View {c}")
            # build table
            for r in range(st_n_bodyparts):
                cols = st.columns(st_n_views + 1)
                # set row titles
                cols[0].text("")
                cols[0].text("")
                cols[0].text(f"Bodypart {r}")
                # set bodypart dropdowns
                for c, col in enumerate(cols[1:]):
                    kp = col.selectbox(
                        f"", st_keypoints, key=f"Bodypart {r} view {c}")
                    st_pcamv_columns[c, r] = np.where(np.array(st_keypoints) == kp)[0]

            print(st_pcamv_columns)
        st.markdown("")

    # construct config file
    if len(st_keypoints) > 1 and st_n_views > 0:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("##### Export project configuration")
        st.markdown("""
            Click on the button below to create a new project; you will then be able to start 
            labeling data and train models!
        """)
        need_update_pcamv = False
        if len(st_pcamv_columns) > 0:
            if len(st_pcamv_columns.flatten()) != len(np.unique(st_pcamv_columns)):
                need_update_pcamv = True
                st.warning(
                    "Duplicate entries in PCA Multiview selections; each entry should be unique")

        update_cfg = st.button("Create project", disabled=need_update_pcamv)

        if update_cfg:
            st_new_vals = {"data": {}}
            st_new_vals["data"]["video_dir"] = "unlabeled_videos"
            st_new_vals["data"]["csv_file"] = "CollectedData.csv"
            st_new_vals["data"]["num_keypoints"] = st_n_keypoints

            if len(st_pcasv_columns) > 0:
                st_new_vals["data"]["columns_for_singleview_pca"] = st_pcasv_columns
            else:
                st_new_vals["data"]["columns_for_singleview_pca"] = None

            if len(st_pcamv_columns) > 0:
                st_new_vals["data"]["mirrored_column_matches"] = [
                    list(t) for t in st_pcamv_columns]
            else:
                st_new_vals["data"]["mirrored_column_matches"] = None


if __name__ == "__main__":
    run()
